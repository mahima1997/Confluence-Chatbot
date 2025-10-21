#!/usr/bin/env python3
"""
Simple launcher for the Confluence Document Chatbot (FastAPI)
"""

import subprocess
import sys
import os

def main():
    """Launch the FastAPI chatbot application"""
    print("🚀 Starting Confluence Document Chatbot (FastAPI)...")
    print("📂 Looking for documents in ../api_data/")

    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        import sklearn
        import fitz  # PyMuPDF
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("🔧 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")

    # Check if API data exists
    api_data_path = os.path.join(os.path.dirname(__file__), "..", "api_data")
    if not os.path.exists(api_data_path):
        print(f"❌ API data not found at {api_data_path}")
        print("💡 Make sure you've processed your PDF files first")
        return

    pages_dir = os.path.join(api_data_path, "pages")
    if not os.path.exists(pages_dir):
        print(f"❌ No processed documents found in {pages_dir}")
        return

    # Count documents
    json_files = [f for f in os.listdir(pages_dir) if f.endswith('.json')]
    print(f"📚 Found {len(json_files)} processed documents")

    if len(json_files) == 0:
        print("❌ No documents found. Make sure to process your PDFs first.")
        return

    # Launch the FastAPI application
    print("🌐 Starting FastAPI server...")
    print("📍 Open http://localhost:5001 in your browser")
    print("📖 API docs available at http://localhost:5001/docs")
    print("❌ Press Ctrl+C to stop the server")

    try:
        # Use uvicorn to run the FastAPI app
        import uvicorn
        uvicorn.run(
            "web_chatbot:app",
            host="0.0.0.0",
            port=5001,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped. Goodbye!")
    except ImportError:
        # Fallback to subprocess if uvicorn import fails
        subprocess.run([sys.executable, "-m", "uvicorn", "web_chatbot:app", "--host", "0.0.0.0", "--port", "5000", "--reload"], cwd=os.path.dirname(__file__))

if __name__ == "__main__":
    main()
