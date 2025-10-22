# Multimodal RAG API

A FastAPI-based web service for querying the Multimodal Retrieval-Augmented Generation system that processes PDF documents and provides intelligent search capabilities across text and table data.

## 🚀 Quick Start

### 1. Activate the Conda Environment
```bash
conda activate confluence_chatbot
```

### 2. Start the API Server
```bash
cd /Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot
python3 rag_api.py
```

The API will be available at `http://localhost:8000`

### 3. Test the API
```bash
# Test with the provided test script
python3 test_api.py

# Or test manually with curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Optimus?"}'
```

## 📡 API Endpoints

### Health Check
- **GET** `/health`
- Check if the API and RAG system are operational

### Main Query
- **POST** `/query`
- Submit queries to the RAG system
- **Body**: `{"query": "your question", "top_k_text": 5, "top_k_table": 3}`

### Simple Search
- **GET** `/search?q=your+query&text_k=5&table_k=3`
- Quick search endpoint (GET request)

### Rebuild Indexes
- **POST** `/rebuild-indexes`
- Rebuild the FAISS indexes (useful when new data is added)

### Documentation
- **GET** `/docs` - Interactive Swagger UI
- **GET** `/redoc` - ReDoc documentation

## 🔧 Configuration

The API automatically:
- ✅ Loads existing RAG indexes if available
- ✅ Builds indexes from JSON files if needed
- ✅ Uses lazy initialization for optimal performance
- ✅ Provides detailed logging and error handling

## 📊 Response Format

### Query Response
```json
{
  "query": "What is Optimus?",
  "text_results": [
    {
      "score": 0.85,
      "text": "Optimus is a...",
      "source_file": "document.json",
      "index": 5
    }
  ],
  "table_results": [
    {
      "score": 0.78,
      "row_text": "Feature | Description",
      "source_file": "document.json",
      "index": 12,
      "table_idx": 2,
      "cells": ["Feature", "Description"]
    }
  ],
  "answer": "Based on retrieved information...",
  "total_results": 8,
  "processing_time_ms": 125.5
}
```

### Health Response
```json
{
  "status": "healthy",
  "message": "RAG system is operational",
  "indexes_built": true,
  "rag_ready": true
}
```

## 🛠️ Development

### Project Structure
```
/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot/
├── rag_api.py           # Main API application
├── multimodal_rag.py    # Core RAG system
├── test_api.py         # API testing script
├── extracted_json/     # Processed PDF data
├── rag_system/         # FAISS indexes and metadata
└── API_README.md       # This documentation
```

### Dependencies
All dependencies are installed in the `confluence_chatbot` conda environment:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector search
- `pdfplumber`, `PyMuPDF` - PDF processing
- `pytesseract` - OCR for images

### Environment Setup
```bash
# Create and activate environment
conda create -n confluence_chatbot python=3.11
conda activate confluence_chatbot

# Install dependencies
pip install fastapi uvicorn pydantic sentence-transformers faiss-cpu
pip install pdfplumber PyMuPDF PyPDF2 pytesseract pillow opencv-python

# Install tesseract OCR
brew install tesseract
```

## 🎯 Features

- **🔍 Multimodal Search**: Query both text paragraphs and table data
- **⚡ Fast Retrieval**: FAISS vector search with cosine similarity
- **📊 Dual Indexes**: Separate indexes for text and table content
- **🛡️ Error Handling**: Comprehensive error handling and logging
- **📈 Performance Tracking**: Response time monitoring
- **🔄 Auto-Indexing**: Automatic index building from JSON files
- **📚 Rich Documentation**: Interactive API docs via Swagger UI

## 🚀 Usage Examples

### Python Client
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the best practices for Optimus?",
    "top_k_text": 5,
    "top_k_table": 3
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Found {result['total_results']} results")
```

### Command Line (curl)
```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about rejection reasons"}'

# Search
curl "http://localhost:8000/search?q=What+is+Optimus&text_k=3&table_k=2"
```

## 🔧 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   conda activate confluence_chatbot
   pip install <missing_module>
   ```

2. **"Indexes not built" errors**
   ```bash
   curl -X POST http://localhost:8000/rebuild-indexes
   ```

3. **Server not starting**
   ```bash
   # Check if port 8000 is in use
   lsof -i :8000
   # Kill process if needed
   kill -9 <PID>
   ```

4. **OCR not working**
   ```bash
   # Ensure tesseract is installed
   brew install tesseract
   # Check tesseract version
   tesseract --version
   ```

## 📈 Performance

- **Response Time**: Typically 25-150ms per query
- **Memory Usage**: ~500MB for indexes and models
- **Concurrent Users**: Designed for moderate concurrent load
- **Index Size**: Depends on document volume (currently ~50 paragraphs + ~88 table rows)

## 🔒 Security Notes

- Currently runs on localhost (not exposed to network)
- No authentication implemented (for development use)
- Consider adding authentication for production deployment

## 🎉 Next Steps

1. **Web Interface**: Add a frontend for easier interaction
2. **Advanced LLM**: Integrate with local LLMs for better answers
3. **Caching**: Add response caching for frequently asked questions
4. **Authentication**: Add API key authentication for production
5. **Monitoring**: Add metrics and logging for production monitoring

---

**Built with ❤️ using FastAPI, FAISS, and Sentence Transformers**
