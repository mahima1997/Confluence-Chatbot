# Confluence Document Chatbot

A complete solution for converting Confluence pages to API format and building a chatbot that can answer questions about the documents.

## 📁 Project Structure

```
MahimuProject/
├── source_data/           # Original PDF files and processing scripts
│   └── data/             # Downloaded Confluence PDFs
├── api_data/             # Processed API-compatible JSON data
│   ├── pages/           # Individual page JSON files
│   ├── manifest.json    # Overview of all documents
│   └── README.md       # API data documentation
├── chatbot/             # Chatbot application
│   ├── web_chatbot.py  # Flask web application
│   ├── run.py          # Easy launcher script
│   ├── templates/      # HTML templates
│   └── requirements.txt # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Start the Chatbot

```bash
cd chatbot
pip3 install -r requirements.txt
python3 run.py
```

Or directly:

```bash
cd chatbot
python3 web_chatbot.py
```

Then open `http://localhost:5000` in your browser.

**Bonus**: Visit `http://localhost:5000/docs` for interactive API documentation!

### 2. Ask Questions

The chatbot can answer questions like:
- "What are the key priorities in the roadmap?"
- "Tell me about best practices"
- "What resources are available?"

## 🔧 Features

### ✅ Complete PDF Processing
- **Text Extraction** - Extracts all readable text from PDFs
- **Table Detection** - Identifies and preserves table structures
- **Visual Element Detection** - Identifies images, diagrams, and charts
- **API Format Conversion** - Converts to Confluence REST API format

### ✅ Smart Chatbot
- **Similarity Search** - Uses TF-IDF and cosine similarity for relevance
- **Confidence Scoring** - Shows how confident the answer is
- **Source Attribution** - Cites which document provided the answer
- **Visual Awareness** - Notes when documents contain diagrams/images

### ✅ Modern Web Interface
- **Beautiful UI** - Gradient design with smooth animations
- **Real-time Chat** - Instant responses with typing indicators
- **Responsive Design** - Works on desktop and mobile
- **Source Display** - Shows document sources with confidence levels

## 📊 Document Analysis

| Document | Tables | Images | Diagrams | Status |
|----------|--------|--------|----------|---------|
| Copy of Optimus 2025 | 4 | 14 | 1 | ✅ Complete |
| Optimus Best Practices | 0 | 0 | 0 | ✅ Complete |
| Optimus Impact in FY'24 | 1 | 0 | 0 | ✅ Complete |
| Optimus Recommendation Rejection | 4 | 2 | 0 | ✅ Complete |
| Optimus Resources | 2 | 0 | 0 | ✅ Complete |
| Roadmap - Optimus | 6 | 0 | 2 | ✅ Complete |

## 🛠️ Technical Details

### Data Processing Pipeline
1. **PDF Text Extraction** - Uses PyPDF2 for text content
2. **Table Detection** - Uses Camelot for table structure
3. **Visual Analysis** - Uses PyMuPDF for image/diagram detection
4. **API Format Conversion** - Creates Confluence-compatible JSON

### Chatbot Architecture
- **Backend**: FastAPI web application with async endpoints
- **Frontend**: Modern HTML/CSS/JavaScript chat interface
- **ML**: TF-IDF vectorization with cosine similarity scoring
- **Database**: File-based JSON storage for documents
- **Server**: Uvicorn ASGI server with auto-reload

## 🎯 Use Cases

- **Document Q&A** - Ask questions about project documentation
- **Knowledge Base** - Search through technical documentation
- **Research Assistant** - Find information across multiple documents
- **Content Discovery** - Explore document collections intelligently

## 🔮 Future Enhancements

- **LLM Integration** - Add GPT/Claude for better answer generation
- **Vector Search** - Replace TF-IDF with semantic embeddings
- **Multi-turn Conversations** - Remember context across questions
- **Document Upload** - Allow users to add new documents
- **Advanced Filtering** - Filter by document type, date, author

## 📝 API Data Format

Each document is converted to Confluence REST API format with:
- Complete metadata (title, author, dates, permissions)
- Full text content in both storage and atlas formats
- Table structures preserved as HTML
- Visual element metadata for transparency
- Proper linking and navigation structure

## 🏗️ Built With

- **Python** - Core processing and web framework
- **FastAPI** - Modern async web API framework
- **Uvicorn** - ASGI server for FastAPI
- **Scikit-learn** - Machine learning for similarity search
- **PyPDF2** - PDF text extraction
- **Camelot** - PDF table extraction
- **PyMuPDF** - PDF visual element analysis
- **HTML/CSS/JavaScript** - Modern web interface

---

**Ready to explore your documents?** Start the chatbot and ask away! 🤖📚
