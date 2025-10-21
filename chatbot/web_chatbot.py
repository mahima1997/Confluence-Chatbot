#!/usr/bin/env python3
"""
Web-based Confluence Document Chatbot using FastAPI with RAG capabilities
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn
import logging

# Import configuration
from config import config

# RAG Pipeline imports
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG dependencies not available: {e}")
    RAG_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app with configuration
app = FastAPI(
    title=f"{config.APP_NAME} API",
    description="A RAG-powered chatbot for answering questions about Confluence documents",
    version=config.APP_VERSION,
    debug=config.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount templates (static files not needed for this simple interface)
templates = Jinja2Templates(directory="templates")

class ConfluenceChatbot:
    def __init__(self, api_data_path: str = None):
        self.api_data_path = Path(api_data_path or config.API_DATA_PATH)
        self.documents = []
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        self.page_links = {}  # Track links between pages
        self.link_graph = {}  # Graph of page relationships
        self.llm = None  # Quantized LLM instance

        # Validate configuration
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration issues: {issues}")

        self.load_documents()
        self.build_link_graph()
        self.setup_rag_pipeline()
        self.setup_llm()

    def load_documents(self):
        """Load all documents from the pages directory"""
        pages_dir = self.api_data_path / "pages"

        for json_file in pages_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)

                # Extract text content from both storage and atlas formats
                content = ""
                if 'body' in doc_data:
                    for format_type in ['storage', 'atlas_doc_format']:
                        if format_type in doc_data['body']:
                            # Extract text from HTML-like content
                            text = self.extract_text_from_html(doc_data['body'][format_type]['value'])
                            if text:
                                content += text + "\n"

                if content.strip():
                    self.documents.append({
                        'id': doc_data.get('id', ''),
                        'title': doc_data.get('title', json_file.stem),
                        'content': content.strip(),
                        'url': doc_data.get('_links', {}).get('webui', ''),
                        'filename': json_file.name,
                        'metadata': {
                            'page_id': doc_data.get('id', ''),
                            'space_key': doc_data.get('space', {}).get('key', ''),
                            'created': doc_data.get('version', {}).get('when', ''),
                            'author': doc_data.get('version', {}).get('by', {}).get('displayName', ''),
                            'has_tables': self.detect_tables(content),
                            'has_links': self.detect_links(doc_data),
                            'visual_elements': self.extract_visual_metadata(doc_data)
                        }
                    })

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        logger.info(f"Loaded {len(self.documents)} documents")

    def build_link_graph(self):
        """Build a graph of page relationships based on links"""
        self.page_links = {}
        self.link_graph = {}

        for doc in self.documents:
            doc_id = doc['id']
            links = doc.get('metadata', {}).get('has_links', [])

            # Initialize link tracking for this document
            self.page_links[doc_id] = {
                'outgoing': [],
                'incoming': [],
                'document': doc
            }

            # Extract page relationships from links
            for link in links:
                if 'confluence.target.com' in link.lower():
                    # This is a Confluence link, try to extract page references
                    linked_pages = self._extract_confluence_page_refs(link, doc_id)
                    self.page_links[doc_id]['outgoing'].extend(linked_pages)

        # Build the graph
        for doc_id, link_data in self.page_links.items():
            self.link_graph[doc_id] = {
                'document': link_data['document'],
                'outgoing': link_data['outgoing'],
                'incoming': []
            }

        # Calculate incoming links
        for doc_id, link_data in self.link_graph.items():
            for outgoing_link in link_data['outgoing']:
                if outgoing_link in self.link_graph:
                    self.link_graph[outgoing_link]['incoming'].append(doc_id)

        logger.info(f"Built link graph with {len(self.link_graph)} nodes")

    def _extract_confluence_page_refs(self, link: str, source_doc_id: str) -> List[str]:
        """Extract page IDs from Confluence links"""
        linked_pages = []

        # Pattern to match Confluence page URLs and extract page IDs
        patterns = [
            r'confluence\.target\.com/display/[^/]+/(.+)',
            r'confluence\.target\.com/pages/(\d+)',
            r'/display/[^/]+/(.+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, link, re.IGNORECASE)
            for match in matches:
                # Look for the page ID in our documents
                for doc in self.documents:
                    if (doc['id'] == match or
                        match.lower() in doc['title'].lower() or
                        doc['title'].lower() in match.lower()):
                        if doc['id'] != source_doc_id:  # Avoid self-references
                            linked_pages.append(doc['id'])

        return linked_pages

    def find_related_documents(self, doc_id: str, max_depth: int = 2) -> List[Dict]:
        """Find documents related through links"""
        if doc_id not in self.link_graph:
            return []

        related_docs = []
        visited = {doc_id}
        queue = [(doc_id, 0)]  # (doc_id, depth)

        while queue:
            current_id, depth = queue.pop(0)

            if depth > max_depth:
                continue

            current_node = self.link_graph[current_id]

            # Add related documents
            for linked_id in current_node['outgoing']:
                if linked_id not in visited:
                    visited.add(linked_id)
                    related_doc = self.link_graph[linked_id]['document']
                    related_doc['relation_type'] = 'linked'
                    related_doc['relation_depth'] = depth + 1
                    related_docs.append(related_doc)
                    queue.append((linked_id, depth + 1))

        return related_docs

    def detect_tables(self, content: str) -> bool:
        """Detect if content contains tables"""
        return 'table' in content.lower() or '|' in content

    def detect_links(self, doc_data: Dict) -> List[str]:
        """Extract links from document"""
        links = []
        try:
            # Look for links in the content
            content = ""
            if 'body' in doc_data:
                for format_type in ['storage', 'atlas_doc_format']:
                    if format_type in doc_data['body']:
                        content += doc_data['body'][format_type]['value']

            # Extract URLs and confluence links
            url_pattern = r'https?://[^\s<>"]+|confluence\.target\.com[^\s<>"]*'
            links = re.findall(url_pattern, content, re.IGNORECASE)
        except:
            pass
        return links

    def extract_visual_metadata(self, doc_data: Dict) -> Dict:
        """Extract comprehensive visual element metadata"""
        visual_data = {
            'has_images': False,
            'has_diagrams': False,
            'has_charts': False,
            'has_tables': False,
            'image_count': 0,
            'diagram_count': 0,
            'chart_count': 0,
            'table_count': 0,
            'visual_types': [],
            'complexity_score': 0
        }

        try:
            # Check PDF export metadata first
            if 'metadata' in doc_data and 'pdf_export' in doc_data['metadata']:
                pdf_export = doc_data['metadata']['pdf_export']
                if pdf_export.get('has_visuals', False):
                    visuals = pdf_export.get('visual_elements', {})

                    # Count different visual types
                    visual_data['image_count'] = visuals.get('images', 0)
                    visual_data['diagram_count'] = visuals.get('diagrams', 0)
                    visual_data['chart_count'] = visuals.get('charts', 0)
                    visual_data['table_count'] = visuals.get('tables', 0)

                    # Set boolean flags
                    visual_data['has_images'] = visual_data['image_count'] > 0
                    visual_data['has_diagrams'] = visual_data['diagram_count'] > 0
                    visual_data['has_charts'] = visual_data['chart_count'] > 0
                    visual_data['has_tables'] = visual_data['table_count'] > 0

                    # Determine visual types present
                    if visual_data['has_images']:
                        visual_data['visual_types'].append('images')
                    if visual_data['has_diagrams']:
                        visual_data['visual_types'].append('diagrams')
                    if visual_data['has_charts']:
                        visual_data['visual_types'].append('charts')
                    if visual_data['has_tables']:
                        visual_data['visual_types'].append('tables')

            # Enhanced detection from content text
            content = ""
            if 'body' in doc_data:
                for format_type in ['storage', 'atlas_doc_format']:
                    if format_type in doc_data['body']:
                        content += doc_data['body'][format_type]['value']

            # Detect visual elements from content patterns
            visual_patterns = {
                'diagrams': [
                    r'diagram', r'flowchart', r'architecture', r'component diagram',
                    r'data flow', r'process flow', r'workflow', r'system design'
                ],
                'charts': [
                    r'chart', r'graph', r'plot', r'bar chart', r'line chart',
                    r'pie chart', r'histogram', r'scatter plot'
                ],
                'tables': [
                    r'table[^a-z]', r'tabular', r'spreadsheet', r'data table'
                ]
            }

            content_lower = content.lower()
            for visual_type, patterns in visual_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower):
                        if visual_type not in visual_data['visual_types']:
                            visual_data['visual_types'].append(visual_type)
                            if visual_type == 'diagrams':
                                visual_data['has_diagrams'] = True
                                visual_data['diagram_count'] += 1
                            elif visual_type == 'charts':
                                visual_data['has_charts'] = True
                                visual_data['chart_count'] += 1
                            elif visual_type == 'tables':
                                visual_data['has_tables'] = True
                                visual_data['table_count'] += 1

            # Calculate complexity score based on visual elements
            complexity_weights = {
                'diagrams': 3,
                'charts': 2,
                'tables': 2,
                'images': 1
            }

            for visual_type in visual_data['visual_types']:
                count = getattr(visual_data, f'{visual_type[:-1]}_count', 0)
                weight = complexity_weights.get(visual_type, 1)
                visual_data['complexity_score'] += count * weight

        except Exception as e:
            logger.warning(f"Error extracting visual metadata: {e}")

        return visual_data

    def setup_rag_pipeline(self):
        """Set up the RAG pipeline with vector search"""
        if not RAG_AVAILABLE:
            logger.warning("RAG dependencies not available, using fallback TF-IDF search")
            return

        try:
            # Initialize text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
            )

            # Initialize embeddings model (configurable)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDINGS_MODEL,
                model_kwargs={'device': 'cpu'}
            )

            # Create vector store from documents (limit for performance)
            if self.documents:
                # Limit documents if configured
                docs_to_process = self.documents[:config.MAX_DOCUMENTS]

                documents = []
                for doc in docs_to_process:
                    # Split document into chunks
                    chunks = self.text_splitter.split_text(doc['content'])
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                'doc_id': doc['id'],
                                'title': doc['title'],
                                'url': doc['url'],
                                'chunk_index': i,
                                **doc['metadata']
                            }
                        ))

                # Create FAISS vector store
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created vector store with {len(documents)} document chunks from {len(docs_to_process)} documents")

        except Exception as e:
            logger.error(f"Failed to setup RAG pipeline: {e}")
            self.vectorstore = None

    def setup_llm(self):
        """Set up the quantized LLM for answer generation"""
        try:
            from llama_cpp import Llama

            # Use absolute path to the model file
            model_path = Path(__file__).parent.parent / "models" / config.LLM_MODEL

            if model_path.exists():
                # Initialize the quantized model with configuration values
                self.llm = Llama(
                    model_path=str(model_path),
                    n_ctx=config.LLM_CONTEXT_WINDOW,
                    n_threads=config.LLM_THREADS,
                    verbose=False  # Reduce logging noise
                )
                logger.info(f"Loaded quantized LLM: {model_path}")
            else:
                logger.warning(f"Quantized model not found at {model_path}, using template-based generation")
                self.llm = None

        except ImportError:
            logger.warning("llama-cpp-python not available, using template-based generation")
            self.llm = None
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.llm = None

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract readable text from HTML content"""
        if not html_content:
            return ""

        # Remove HTML tags but keep the text
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\-.,!?()]', ' ', text)

        return text.strip()

    def find_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find most relevant documents for a query using RAG or fallback TF-IDF"""
        if not self.documents:
            return []

        # Use RAG pipeline if available
        if self.vectorstore:
            try:
                # Search for relevant chunks
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k*2)

                # Group by document and calculate aggregated scores
                doc_scores = {}
                for doc, score in docs_with_scores:
                    doc_id = doc.metadata['doc_id']
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'score': score,
                            'chunks': [],
                            'document': None
                        }
                    doc_scores[doc_id]['chunks'].append(doc)
                    # Use the best score for the document
                    if score < doc_scores[doc_id]['score']:
                        doc_scores[doc_id]['score'] = score

                # Find the original documents
                relevant_docs = []
                for doc_id, data in doc_scores.items():
                    original_doc = next((d for d in self.documents if d['id'] == doc_id), None)
                    if original_doc:
                        # Combine relevant chunks for context
                        relevant_content = " ".join([chunk.page_content for chunk in data['chunks'][:2]])
                        original_doc = original_doc.copy()
                        original_doc['relevant_content'] = relevant_content
                        original_doc['similarity'] = 1.0 / (1.0 + data['score'])  # Convert distance to similarity
                        relevant_docs.append(original_doc)

                # Sort by similarity and return top k
                relevant_docs.sort(key=lambda x: x['similarity'], reverse=True)
                return relevant_docs[:top_k]

            except Exception as e:
                logger.error(f"RAG search failed: {e}")
                # Fall back to TF-IDF

        # Fallback TF-IDF search
        return self._tfidf_search(query, top_k)

    def _tfidf_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Fallback TF-IDF search when RAG is not available"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Prepare documents for similarity search
        doc_contents = [doc['content'] for doc in self.documents]
        doc_contents.append(query)  # Add query for comparison

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(doc_contents)

        # Calculate cosine similarity
        query_vector = tfidf_matrix[-1]  # Last item is the query
        doc_vectors = tfidf_matrix[:-1]  # All others are documents

        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only include relevant results
                doc = self.documents[idx].copy()
                doc['similarity'] = float(similarities[idx])
                results.append(doc)

        return results

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG pipeline with cross-page linking"""
        relevant_docs = self.find_relevant_documents(question, top_k=3)

        if not relevant_docs:
            return {
                'answer': "I couldn't find relevant information to answer your question in the available documents.",
                'sources': [],
                'confidence': 0.0
            }

        # Generate answer using RAG approach
        best_match = relevant_docs[0]
        confidence = best_match['similarity']

        # Enhanced answer generation with cross-references
        context = self._build_context(relevant_docs, question)
        answer = self._generate_answer(context, question)

        # Find related documents through links for additional context
        related_docs = []
        for doc in relevant_docs[:2]:  # Check top 2 documents for links
            doc_related = self.find_related_documents(doc['id'], max_depth=1)
            related_docs.extend(doc_related)

        # Remove duplicates and add to context if relevant
        seen_ids = {doc['id'] for doc in relevant_docs}
        unique_related_docs = [doc for doc in related_docs if doc['id'] not in seen_ids]

        if unique_related_docs:
            # Add related documents to context
            enhanced_context = context + "\n\nRelated Documents:\n" + self._build_context(unique_related_docs[:2], question)
            answer = self._generate_answer(enhanced_context, question)

        # Add visual and link information
        enhanced_answer = self._enhance_answer_with_metadata(answer, relevant_docs + unique_related_docs)

        # Format all sources (primary + related)
        all_sources = self._format_sources(relevant_docs + unique_related_docs)

        # Structure the response for better frontend display
        structured_response = {
            'answer': enhanced_answer,
            'sources': all_sources,
            'confidence': float(confidence),
            'rag_enabled': self.vectorstore is not None,
            'cross_references': int(len(unique_related_docs)),
            'best_practices': self._extract_best_practices(context, question),
            'document_relationships': self._get_document_relationships(relevant_docs + unique_related_docs)
        }

        return structured_response

    def _extract_best_practices(self, context: str, question: str) -> List[Dict]:
        """Extract structured best practices from context for better display"""
        best_practices = []

        # Look for numbered practices in the context
        import re

        # Pattern to match numbered practices like "1. Practice: Description (Source: ...)"
        practice_pattern = r'(\d+)\.\s*([^:]+):\s*([^()]+)(?:\s*\(Source:\s*([^)]+)\))?'

        matches = re.findall(practice_pattern, context, re.MULTILINE | re.DOTALL)

        for match in matches:
            number, title, description, source = match

            # Clean up the description
            description = description.strip()
            if description.endswith('.'):
                description = description[:-1]

            best_practices.append({
                'number': int(number),
                'title': title.strip(),
                'description': description,
                'source': source.strip() if source else "General context"
            })

        return best_practices

    def _get_document_relationships(self, relevant_docs: List[Dict]) -> Dict:
        """Get document relationship data for graph visualization"""
        nodes = []
        edges = []

        # Create nodes for each document
        for doc in relevant_docs:
            # Determine node type based on visual elements
            visual_badges = []
            visuals = doc.get('metadata', {}).get('visual_elements', {})

            if visuals.get('has_diagrams'):
                visual_badges.append('📊')
            if visuals.get('has_images'):
                visual_badges.append('🖼️')
            if visuals.get('has_tables'):
                visual_badges.append('📋')

            node = {
                'id': doc['id'],
                'title': doc['title'],
                'type': 'primary' if doc.get('relation_type') == 'primary' else 'related',
                'visual_elements': visual_badges,
                'link_count': len(doc.get('metadata', {}).get('has_links', [])),
                'similarity': doc.get('similarity', 0)
            }
            nodes.append(node)

            # Add edges for document relationships
            if doc.get('relation_type') == 'linked':
                # Find the primary document this links from
                for primary_doc in relevant_docs:
                    if primary_doc.get('relation_type') == 'primary':
                        edges.append({
                            'from': primary_doc['id'],
                            'to': doc['id'],
                            'type': 'references',
                            'label': f"Linked (depth {doc.get('relation_depth', 1)})"
                        })

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _build_context(self, relevant_docs: List[Dict], question: str) -> str:
        """Build context for answer generation"""
        context_parts = []

        for i, doc in enumerate(relevant_docs):
            # Use relevant content if available (from RAG), otherwise use document content
            content = doc.get('relevant_content', doc['content'][:800])

            context_parts.append(f"Document {i+1}: {doc['title']}\nContent: {content}")

            # Add metadata context
            metadata = doc.get('metadata', {})
            if metadata.get('has_tables'):
                context_parts.append(f"(This document contains tables)")
            if metadata.get('visual_elements', {}).get('has_images'):
                context_parts.append(f"(This document contains images)")
            if metadata.get('visual_elements', {}).get('has_diagrams'):
                context_parts.append(f"(This document contains diagrams)")

        return "\n\n".join(context_parts)

    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using quantized LLM or fallback to template-based approach"""
        if not context.strip():
            return "I found some documents but couldn't extract meaningful content to answer your question."

        # Use quantized LLM if available
        if self.llm:
            try:
                # Create a LLaMA 2 chat-format prompt
                system_prompt = "You are a helpful assistant that answers questions based on provided context from documents. Be accurate, concise, and cite your sources when possible."

                user_prompt = f"""Context from documents:
{context[:1200]}  # Limit context to avoid token overflow

Question: {question}

Please provide a helpful answer based only on the context above."""

                # Use LLaMA 2 chat format
                prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

                # Generate response with better parameters
                response = self.llm(
                    prompt,
                    max_tokens=min(config.LLM_MAX_TOKENS, 512),  # Limit to 512 tokens for safety
                    temperature=0.3,  # Slightly higher for better responses
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["</s>", "[INST]", "[/INST]"]  # Proper LLaMA 2 stop tokens
                )

                answer_text = response['choices'][0]['text'].strip()

                # Clean up the response (remove any remaining prompt artifacts)
                answer_text = answer_text.replace("[/INST]", "").replace("</s>", "").strip()

                if not answer_text or len(answer_text) < 10:  # Minimum response length
                    logger.warning(f"LLM response too short or empty: '{answer_text}'")
                    raise ValueError("Insufficient response from LLM")

                logger.info(f"LLM generated response: {len(answer_text)} characters")
                return answer_text

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, falling back to template-based approach")

        # Fallback to template-based generation
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []

        question_words = set(word.lower() for word in question.split() if len(word) > 3)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                sentence_words = set(word.lower() for word in sentence.split())
                # Check if sentence shares meaningful words with question
                common_words = question_words.intersection(sentence_words)
                if len(common_words) >= 1:
                    relevant_sentences.append(sentence)

        if relevant_sentences:
            # Use the most relevant sentences
            answer_text = '. '.join(relevant_sentences[:3])
            if len(answer_text) > 500:
                answer_text = answer_text[:497] + '...'
        else:
            # Fallback to first part of context
            answer_text = context[:400] + '...'

        return f"Based on the available documentation: {answer_text}"

    def _enhance_answer_with_metadata(self, answer: str, relevant_docs: List[Dict]) -> str:
        """Enhance answer with comprehensive visual and link metadata"""
        enhancements = []

        for doc in relevant_docs[:3]:  # Check top 3 documents for metadata
            metadata = doc.get('metadata', {})
            visuals = metadata.get('visual_elements', {})

            # Enhanced visual information
            visual_descriptions = []
            if visuals.get('has_diagrams') and visuals.get('diagram_count', 0) > 0:
                visual_descriptions.append(f"{visuals['diagram_count']} diagram{'' if visuals['diagram_count'] == 1 else 's'}")
            if visuals.get('has_charts') and visuals['chart_count'] > 0:
                visual_descriptions.append(f"{visuals['chart_count']} chart{'' if visuals['chart_count'] == 1 else 's'}")
            if visuals.get('has_images') and visuals['image_count'] > 0:
                visual_descriptions.append(f"{visuals['image_count']} image{'' if visuals['image_count'] == 1 else 's'}")
            if visuals.get('has_tables') and visuals['table_count'] > 0:
                visual_descriptions.append(f"{visuals['table_count']} table{'' if visuals['table_count'] == 1 else 's'}")

            if visual_descriptions:
                visual_text = ', '.join(visual_descriptions)
                complexity = visuals.get('complexity_score', 0)
                if complexity > 5:
                    visual_text += " (complex visual content)"
                enhancements.append(f"Document '{doc['title']}' contains: {visual_text}")

            # Enhanced link information
            links = metadata.get('has_links', [])
            if links:
                link_text = f"Document '{doc['title']}' references {len(links)} external link{'' if len(links) == 1 else 's'}"
                enhancements.append(link_text)

            # Add relationship context for linked documents
            if doc.get('relation_type') == 'linked':
                depth = doc.get('relation_depth', 1)
                enhancements.append(f"Document '{doc['title']}' is linked (depth {depth}) from primary sources")

        if enhancements:
            answer += f"\n\nAdditional context: {'; '.join(enhancements)}"

        return answer

    def _format_sources(self, relevant_docs: List[Dict]) -> List[Dict]:
        """Format sources for response"""
        sources = []

        for doc in relevant_docs:
            source_info = {
                    'title': doc['title'],
                'url': doc.get('url', ''),
                'similarity': float(round(doc.get('similarity', 0.0), 3)),  # Convert to float
                'page_id': doc.get('id', ''),
                'relation_type': doc.get('relation_type', 'primary'),
                'relation_depth': int(doc.get('relation_depth', 0)),  # Convert to int
                'metadata': doc.get('metadata', {})
            }

            # Add visual information
            visuals = doc.get('metadata', {}).get('visual_elements', {})
            if visuals.get('has_images') or visuals.get('has_diagrams'):
                visual_types = []
                if visuals.get('has_images'):
                    visual_types.append('images')
                if visuals.get('has_diagrams'):
                    visual_types.append('diagrams')
                source_info['visual_elements'] = visual_types

            # Add link information
            links = doc.get('metadata', {}).get('has_links', [])
            if links:
                source_info['link_count'] = int(len(links))  # Convert to int

            sources.append(source_info)

        return sources

    def get_document_data(self, filename: str):
        """Get full document data by filename"""
        try:
            json_file = self.api_data_path / "pages" / filename
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def document_has_visuals(self, filename: str) -> bool:
        """Check if a document has visual elements"""
        doc_data = self.get_document_data(filename)
        if doc_data and 'metadata' in doc_data and 'pdf_export' in doc_data['metadata']:
            return doc_data['metadata']['pdf_export'].get('has_visuals', False)
        return False

# Initialize chatbot
chatbot = ConfluenceChatbot()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    """Answer a question about the documents"""
    try:
        data = await request.json()
        question = data.get('question', '').strip()

        if not question:
            raise HTTPException(status_code=400, detail="No question provided")

        result = chatbot.answer_question(question)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents_loaded": len(chatbot.documents)}

if __name__ == "__main__":
    # Validate configuration before starting
    issues = config.validate()
    if issues:
        logger.error(f"Configuration validation failed: {issues}")
        exit(1)

    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"API data path: {config.API_DATA_PATH}")
    logger.info(f"Documents: {len(chatbot.documents)}")
    logger.info(f"RAG enabled: {chatbot.vectorstore is not None}")

    uvicorn.run(
        "web_chatbot:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower(),
        timeout_keep_alive=config.TIMEOUT
    )
