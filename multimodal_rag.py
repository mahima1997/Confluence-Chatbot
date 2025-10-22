#!/usr/bin/env python3
"""
multimodal_rag.py

- Load Confluence-like JSON files produced by extract_pdfs_to_confluence_json.py
- Build two FAISS vector indexes:
    1) text_index (paragraph-level)
    2) table_index (each table row as a document)
- On query:
    - retrieve top-k from both indexes separately
    - construct a retrieval context
    - generate an answer using a local LLM or fallback to rule-based responses
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import sys

# ------------------ CONFIG ------------------
# Base project directory
BASE_DIR = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot")
JSON_DIR = BASE_DIR / "extracted_json"
RAG_DIR = BASE_DIR / "rag_system"
RAG_DIR.mkdir(parents=True, exist_ok=True)

# Embedding models - different models for different content types
TEXT_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"     # small, fast for text (384 dims)
TABLE_EMBED_MODEL_NAME = "all-mpnet-base-v2"   # more powerful for tables (768 dims)

# Index paths
TEXT_INDEX_PATH = RAG_DIR / "text_index.faiss"
TABLE_INDEX_PATH = RAG_DIR / "table_index.faiss"
METADATA_PATH = RAG_DIR / "metadata.json"

# Retrieval settings
TOP_K_TEXT = 5
TOP_K_TABLE = 3
EMBED_BATCH = 32

# Generation settings (using rule-based responses for now)
USE_LOCAL_LLM = False  # Set to True if you have llama-cpp-python setup
# --------------------------------------------

class MultimodalRAG:
    def __init__(self, json_dir: Path = JSON_DIR):
        self.json_dir = json_dir
        self.text_embed_model: Optional[SentenceTransformer] = None
        self.table_embed_model: Optional[SentenceTransformer] = None
        self.text_index: Optional[faiss.Index] = None
        self.table_index: Optional[faiss.Index] = None
        self.metadata: Dict = {}

    def load_json_files(self) -> List[Dict]:
        """Load all JSON files from the extracted_json directory"""
        json_files = list(self.json_dir.glob("*.json"))
        payloads = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    payload["_source_file"] = json_file.name  # Track source
                    payloads.append(payload)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        return payloads

    def extract_text_and_tables(self, payloads: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract paragraphs and table rows from all JSON payloads"""
        all_paragraphs = []
        all_table_rows = []

        for payload_idx, payload in enumerate(payloads):
            source_file = payload.get("_source_file", f"payload_{payload_idx}")

            # Extract from body.storage.value (HTML-like content)
            body_html = payload.get("body", {}).get("storage", {}).get("value", "")

            # Extract paragraphs (text between HTML tags)
            paragraphs = self._extract_paragraphs_from_html(body_html, source_file, payload_idx)

            # Extract tables and their rows
            table_rows = self._extract_table_rows_from_html(body_html, source_file, payload_idx)

            all_paragraphs.extend(paragraphs)
            all_table_rows.extend(table_rows)

        return all_paragraphs, all_table_rows

    def _extract_paragraphs_from_html(self, html: str, source_file: str, payload_idx: int) -> List[Dict]:
        """Extract paragraph text from HTML content"""
        paragraphs = []

        # Remove table blocks first to avoid duplication
        table_pattern = re.compile(r"(<table.*?>.*?</table>)", flags=re.DOTALL|re.IGNORECASE)
        html_no_tables = table_pattern.sub(" ", html)

        # Extract text from p, h1-h6 tags
        para_pattern = re.compile(r"<(p|h[1-6])[^>]*>(.*?)</\1>", flags=re.DOTALL|re.IGNORECASE)

        for match in para_pattern.finditer(html_no_tables):
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if text and len(text) > 20:  # Filter very short texts
                paragraphs.append({
                    "text": text,
                    "source_file": source_file,
                    "payload_idx": payload_idx,
                    "type": "paragraph"
                })

        return paragraphs

    def _extract_table_rows_from_html(self, html: str, source_file: str, payload_idx: int) -> List[Dict]:
        """Extract table rows from HTML content"""
        table_rows = []

        # Find all table blocks
        table_pattern = re.compile(r"(<table.*?>.*?</table>)", flags=re.DOTALL|re.IGNORECASE)
        tables = table_pattern.findall(html)

        for table_idx, table_html in enumerate(tables):
            # Extract rows
            row_pattern = re.compile(r"<tr>(.*?)</tr>", flags=re.DOTALL|re.IGNORECASE)

            # Extract table headers (first row with th elements)
            header_cells = []
            first_row_html = None

            # Find the first row
            rows = row_pattern.findall(table_html)
            if rows:
                first_row_html = rows[0]
                # Look for th (header) cells in first row
                th_pattern = re.compile(r"<th[^>]*>(.*?)</th>", flags=re.DOTALL|re.IGNORECASE)
                header_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in th_pattern.findall(first_row_html)]

            # If no th cells found, check if first row has td cells that look like headers
            if not header_cells and rows:
                td_pattern = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.DOTALL|re.IGNORECASE)
                potential_headers = [re.sub(r"<[^>]+>", "", cell).strip() for cell in td_pattern.findall(first_row_html)]
                # Assume first row is headers if cells are short and look like column names
                if potential_headers and all(len(cell) < 50 and not any(char.isdigit() for char in cell) for cell in potential_headers):
                    header_cells = potential_headers

            # Process data rows (skip header row)
            data_rows = rows[1:] if header_cells else rows

            for row_idx, row_match in enumerate(data_rows):
                row_html = row_match

                # Extract cells (td or th)
                cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", flags=re.DOTALL|re.IGNORECASE)
                cells = cell_pattern.findall(row_html)

                if cells:
                    # Clean cell content
                    clean_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
                    row_text = " | ".join(clean_cells)

                    if row_text.strip() and len(row_text) > 5:  # Filter empty/short rows
                        table_rows.append({
                            "row_text": row_text,
                            "cells": clean_cells,
                            "source_file": source_file,
                            "payload_idx": payload_idx,
                            "table_idx": table_idx,
                            "row_idx": row_idx,
                            "table_headers": header_cells,
                            "header_text": " | ".join(header_cells) if header_cells else "",
                            "type": "table_row"
                        })

        return table_rows

    def embed_texts(self, texts: List[str], model_name: str, batch_size: int = EMBED_BATCH) -> np.ndarray:
        """Embed texts using specified sentence transformer model"""
        # Initialize appropriate model
        if model_name == TEXT_EMBED_MODEL_NAME:
            if not self.text_embed_model:
                self.text_embed_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)
            embed_model = self.text_embed_model
        elif model_name == TABLE_EMBED_MODEL_NAME:
            if not self.table_embed_model:
                self.table_embed_model = SentenceTransformer(TABLE_EMBED_MODEL_NAME)
            embed_model = self.table_embed_model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.zeros((0, embed_model.get_sentence_embedding_dimension()))

    def build_indexes(self):
        """Build FAISS indexes for text and table data"""
        print("Loading JSON files...")
        payloads = self.load_json_files()

        if not payloads:
            raise ValueError(f"No JSON files found in {self.json_dir}")

        print(f"Found {len(payloads)} JSON files")

        print("Extracting text and tables...")
        paragraphs, table_rows = self.extract_text_and_tables(payloads)

        print(f"Extracted {len(paragraphs)} paragraphs and {len(table_rows)} table rows")

        if len(paragraphs) == 0 and len(table_rows) == 0:
            raise ValueError("No content extracted from JSON files")

        # Prepare text data for embedding
        paragraph_texts = [p["text"] for p in paragraphs]
        table_texts = [t["row_text"] for t in table_rows]

        print("Generating embeddings...")

        # Embed paragraphs using text model
        paragraph_embeddings = self.embed_texts(paragraph_texts, TEXT_EMBED_MODEL_NAME).astype("float32")

        # Embed table rows using table model
        table_embeddings = self.embed_texts(table_texts, TABLE_EMBED_MODEL_NAME).astype("float32")

        print(f"Paragraph embeddings shape: {paragraph_embeddings.shape}")
        print(f"Table embeddings shape: {table_embeddings.shape}")

        # Build FAISS indexes (normalized for cosine similarity)
        if paragraph_embeddings.shape[0] > 0:
            print("Building text index...")
            faiss.normalize_L2(paragraph_embeddings)
            self.text_index = faiss.IndexFlatIP(paragraph_embeddings.shape[1])
            self.text_index.add(paragraph_embeddings)
            faiss.write_index(self.text_index, str(TEXT_INDEX_PATH))

        if table_embeddings.shape[0] > 0:
            print("Building table index...")
            faiss.normalize_L2(table_embeddings)
            self.table_index = faiss.IndexFlatIP(table_embeddings.shape[1])
            self.table_index.add(table_embeddings)
            faiss.write_index(self.table_index, str(TABLE_INDEX_PATH))

        # Save metadata with model information
        self.metadata = {
            "paragraphs": paragraphs,
            "table_rows": table_rows,
            "text_embed_model": TEXT_EMBED_MODEL_NAME,
            "table_embed_model": TABLE_EMBED_MODEL_NAME,
            "text_embed_dim": paragraph_embeddings.shape[1] if paragraph_embeddings.shape[0] > 0 else 0,
            "table_embed_dim": table_embeddings.shape[1] if table_embeddings.shape[0] > 0 else 0,
            "num_paragraphs": len(paragraphs),
            "num_table_rows": len(table_rows)
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print("✅ Indexes built and saved!")

    def load_indexes(self):
        """Load existing indexes and metadata"""
        if not METADATA_PATH.exists():
            raise FileNotFoundError("Metadata not found. Run build_indexes() first.")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if TEXT_INDEX_PATH.exists():
            self.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
        else:
            self.text_index = None

        if TABLE_INDEX_PATH.exists():
            self.table_index = faiss.read_index(str(TABLE_INDEX_PATH))
        else:
            self.table_index = None

        # Initialize both embedding models if needed
        if not self.text_embed_model and TEXT_EMBED_MODEL_NAME:
            self.text_embed_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)
        if not self.table_embed_model and TABLE_EMBED_MODEL_NAME:
            self.table_embed_model = SentenceTransformer(TABLE_EMBED_MODEL_NAME)

    def retrieve(self, query: str, top_k_text: int = TOP_K_TEXT, top_k_table: int = TOP_K_TABLE) -> Dict:
        """Retrieve top-k results from both indexes using appropriate models"""
        if not self.text_embed_model and not self.table_embed_model:
            self.load_indexes()

        results = {"text": [], "table": []}

        # Retrieve from text index using text model
        if self.text_index is not None and len(self.metadata.get("paragraphs", [])) > 0:
            # Embed query using text model
            query_text_embedding = self.text_embed_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_text_embedding)

            distances, indices = self.text_index.search(query_text_embedding.astype("float32"), top_k_text)

            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata["paragraphs"]):
                    para = self.metadata["paragraphs"][idx]
                    results["text"].append({
                        "score": float(score),
                        "text": para["text"],
                        "source_file": para["source_file"],
                        "index": int(idx)
                    })

        # Retrieve from table index using table model
        if self.table_index is not None and len(self.metadata.get("table_rows", [])) > 0:
            # Embed query using table model
            query_table_embedding = self.table_embed_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_table_embedding)

            distances, indices = self.table_index.search(query_table_embedding.astype("float32"), top_k_table)

            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata["table_rows"]):
                    row = self.metadata["table_rows"][idx]
                    results["table"].append({
                        "score": float(score),
                        "row_text": row["row_text"],
                        "cells": row["cells"],
                        "source_file": row["source_file"],
                        "table_idx": row["table_idx"],
                        "index": int(idx)
                    })

        return results

    def _query_matches_table_headers(self, query: str, table_headers: List[str]) -> bool:
        """Check if query matches table headers/column names"""
        if not table_headers:
            return False

        query_lower = query.lower()
        header_text = " ".join(table_headers).lower()

        # Check for exact matches or significant overlap
        query_words = set(query_lower.split())
        header_words = set(header_text.split())

        # If query words significantly overlap with header words
        overlap = query_words.intersection(header_words)
        overlap_ratio = len(overlap) / len(query_words) if query_words else 0

        # Return True if overlap is significant (>50%) or if query contains header terms
        return overlap_ratio > 0.3 or any(header_word in query_lower for header_word in header_words if len(header_word) > 3)

    def _get_complete_table_for_headers(self, query: str, table_idx: int, source_file: str) -> List[Dict]:
        """Get complete table rows when query matches table headers"""
        # Find all rows for this table
        table_rows = []
        for row in self.metadata.get("table_rows", []):
            if (row.get("table_idx") == table_idx and
                row.get("source_file") == source_file and
                row.get("table_headers")):

                # Check if this row's headers match the query
                if self._query_matches_table_headers(query, row["table_headers"]):
                    table_rows.append({
                        "score": 1.0,  # High score for header matches
                        "row_text": row["row_text"],
                        "cells": row["cells"],
                        "source_file": row["source_file"],
                        "table_idx": row["table_idx"],
                        "row_idx": row["row_idx"],
                        "table_headers": row["table_headers"],
                        "header_text": row["header_text"],
                        "index": row.get("index", 0),
                        "is_complete_table": True,
                        "matched_by_headers": True
                    })

        return table_rows

    def retrieve(self, query: str, top_k_text: int = TOP_K_TEXT, top_k_table: int = TOP_K_TABLE) -> Dict:
        """Retrieve top-k results from both indexes using appropriate models"""
        if not self.text_embed_model and not self.table_embed_model:
            self.load_indexes()

        results = {"text": [], "table": []}

        # Retrieve from text index using text model
        if self.text_index is not None and len(self.metadata.get("paragraphs", [])) > 0:
            # Embed query using text model
            query_text_embedding = self.text_embed_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_text_embedding)

            distances, indices = self.text_index.search(query_text_embedding.astype("float32"), top_k_text)

            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata["paragraphs"]):
                    para = self.metadata["paragraphs"][idx]
                    results["text"].append({
                        "score": float(score),
                        "text": para["text"],
                        "source_file": para["source_file"],
                        "index": int(idx)
                    })

        # Retrieve from table index using table model
        if self.table_index is not None and len(self.metadata.get("table_rows", [])) > 0:
            # First, check if query matches any table headers - if so, return complete tables
            complete_table_results = []

            # Group table rows by table_idx and source_file
            table_groups = {}
            for row in self.metadata.get("table_rows", []):
                key = (row.get("table_idx", 0), row.get("source_file", ""))
                if key not in table_groups:
                    table_groups[key] = []
                table_groups[key].append(row)

            # Check each table group for header matches
            for (table_idx, source_file), rows in table_groups.items():
                if rows and rows[0].get("table_headers"):
                    if self._query_matches_table_headers(query, rows[0]["table_headers"]):
                        # Return all rows from this table
                        for row in rows:
                            complete_table_results.append({
                                "score": 1.0,  # High score for header matches
                                "row_text": row["row_text"],
                                "cells": row["cells"],
                                "source_file": row["source_file"],
                                "table_idx": row["table_idx"],
                                "row_idx": row["row_idx"],
                                "table_headers": row["table_headers"],
                                "header_text": row["header_text"],
                                "index": row.get("index", 0),
                                "is_complete_table": True,
                                "matched_by_headers": True
                            })

            # If we found complete table results, use them (up to top_k_table)
            if complete_table_results:
                # Sort by score and take top results
                complete_table_results.sort(key=lambda x: x["score"], reverse=True)
                results["table"] = complete_table_results[:top_k_table]
            else:
                # Otherwise, do regular similarity search
                query_table_embedding = self.table_embed_model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(query_table_embedding)

                distances, indices = self.table_index.search(query_table_embedding.astype("float32"), top_k_table)

                for score, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.metadata["table_rows"]):
                        row = self.metadata["table_rows"][idx]
                        results["table"].append({
                            "score": float(score),
                            "row_text": row["row_text"],
                            "cells": row["cells"],
                            "source_file": row["source_file"],
                            "table_idx": row["table_idx"],
                            "index": int(idx)
                        })

        return results

    def generate_answer(self, query: str, retrievals: Dict) -> str:
        """Generate an answer based on retrieved results (rule-based for now)"""
        if not retrievals["text"] and not retrievals["table"]:
            return "I don't have enough information to answer this question based on the available documents."

        # Build context from retrieved results
        context_parts = []

        if retrievals["text"]:
            context_parts.append("**Text passages:**")
            for i, result in enumerate(retrievals["text"], 1):
                context_parts.append(f"{i}. {result['text']} (from {result['source_file']})")

        if retrievals["table"]:
            context_parts.append("\n**Table data:**")
            for i, result in enumerate(retrievals["table"], 1):
                context_parts.append(f"{i}. {result['row_text']} (from {result['source_file']}, table {result['table_idx']})")

        context = "\n".join(context_parts)

        # Simple rule-based answer generation
        if "what" in query.lower() and "optimus" in query.lower():
            # Look for relevant information about Optimus
            optimus_info = []
            for result in retrievals["text"]:
                if "optimus" in result["text"].lower():
                    optimus_info.append(result["text"])

            if optimus_info:
                return f"Based on the retrieved information, here's what I found about Optimus:\n\n" + "\n".join(optimus_info[:2])

        elif "table" in query.lower() or "data" in query.lower():
            # Return table information
            if retrievals["table"]:
                return f"I found relevant table data:\n\n" + "\n".join([
                    f"• {result['row_text']}" for result in retrievals["table"][:3]
                ])

        # Default response
        return f"Based on the retrieved context, I can provide the following information:\n\n{context[:1000]}..."

def main():
    """Main function to test the RAG system"""
    rag = MultimodalRAG()

    # Build indexes if they don't exist
    if not (TEXT_INDEX_PATH.exists() and TABLE_INDEX_PATH.exists() and METADATA_PATH.exists()):
        print("Building RAG indexes...")
        rag.build_indexes()
    else:
        print("Loading existing indexes...")
        rag.load_indexes()

    # Test queries
    test_queries = [
        "What is Optimus?",
        "Tell me about recommendation rejection reasons",
        "What are the best practices for Optimus?",
        "Show me data about campaigns and budget"
    ]

    print("\n" + "="*50)
    print("MULTIMODAL RAG SYSTEM TEST")
    print("="*50)

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)

        # Retrieve results
        results = rag.retrieve(query)

        print(f"📄 Text results ({len(results['text'])}):")
        for i, result in enumerate(results['text'], 1):
            print(f"  {i}. [{result['source_file']}] {result['text'][:100]}... (score: {result['score']:.3f})")

        print(f"\n📊 Table results ({len(results['table'])}):")
        for i, result in enumerate(results['table'], 1):
            print(f"  {i}. [{result['source_file']}] {result['row_text']} (score: {result['score']:.3f})")

        # Generate answer
        answer = rag.generate_answer(query, results)
        print(f"\n🤖 Answer: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
