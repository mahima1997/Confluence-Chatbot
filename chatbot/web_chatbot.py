#!/usr/bin/env python3
"""
Web-based Confluence Document Chatbot using FastAPI
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uvicorn

app = FastAPI(title="Confluence Document Chatbot", description="A chatbot for answering questions about Confluence documents")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ConfluenceChatbot:
    def __init__(self, api_data_path: str = "../api_data"):
        self.api_data_path = Path(api_data_path)
        self.documents = []
        self.load_documents()

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
                        'filename': json_file.name
                    })

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(self.documents)} documents")

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
        """Find most relevant documents for a query"""
        if not self.documents:
            return []

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
        """Answer a question based on the documents"""
        relevant_docs = self.find_relevant_documents(question)

        if not relevant_docs:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0
            }

        # Simple answer generation
        best_match = relevant_docs[0]
        confidence = best_match['similarity']

        # Extract key sentences that might answer the question
        sentences = re.split(r'[.!?]+', best_match['content'])
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word.lower() in sentence.lower() for word in question.split()):
                relevant_sentences.append(sentence)

        # Check if the document has visual elements
        doc_data = self.get_document_data(best_match['filename'])
        has_visuals = False
        visual_note = ""

        if doc_data and 'metadata' in doc_data and 'pdf_export' in doc_data['metadata']:
            pdf_export = doc_data['metadata']['pdf_export']
            if pdf_export.get('has_visuals', False):
                has_visuals = True
                visuals = pdf_export.get('visual_elements', {})
                images = visuals.get('images', 0)
                diagrams = visuals.get('diagrams', 0)

                if images > 0 or diagrams > 0:
                    visual_parts = []
                    if images > 0:
                        visual_parts.append(f"{images} image{'s' if images > 1 else ''}")
                    if diagrams > 0:
                        visual_parts.append(f"{diagrams} diagram{'s' if diagrams > 1 else ''}")

                    visual_note = f" (Note: This document contains {' and '.join(visual_parts)} that may provide additional context not captured in text extraction)"

        base_answer = "Based on the document '{}': {}".format(
            best_match['title'],
            '. '.join(relevant_sentences[:2]) if relevant_sentences else best_match['content'][:300] + '...'
        )

        if has_visuals:
            base_answer += visual_note

        return {
            'answer': base_answer,
            'sources': [
                {
                    'title': doc['title'],
                    'url': doc['url'],
                    'similarity': doc['similarity'],
                    'has_visuals': self.document_has_visuals(doc['filename'])
                }
                for doc in relevant_docs
            ],
            'confidence': float(confidence)
        }

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
    uvicorn.run(
        "web_chatbot:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
