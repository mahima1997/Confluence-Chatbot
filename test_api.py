#!/usr/bin/env python3
"""
test_api.py

Simple test script to interact with the Multimodal RAG API
"""

import requests
import json
import sys
from typing import Dict, Any

# API base URL
API_BASE = "http://localhost:8000"

def test_health() -> Dict[str, Any]:
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.json()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return {"status": "error", "message": str(e)}

def test_query(query: str, top_k_text: int = 3, top_k_table: int = 2) -> Dict[str, Any]:
    """Test the main query endpoint"""
    try:
        payload = {
            "query": query,
            "top_k_text": top_k_text,
            "top_k_table": top_k_table,
            "include_context": True
        }

        response = requests.post(f"{API_BASE}/query", json=payload)
        return response.json()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {"error": str(e)}

def test_search(query: str, text_k: int = 3, table_k: int = 2) -> list:
    """Test the search endpoint"""
    try:
        params = {
            "q": query,
            "text_k": text_k,
            "table_k": table_k
        }

        response = requests.get(f"{API_BASE}/search", params=params)
        return response.json()
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

def main():
    """Main test function"""
    print("🧪 Testing Multimodal RAG API")
    print("=" * 50)

    # Test health check
    print("📋 Testing health check...")
    health = test_health()
    print(f"Status: {health.get('status', 'unknown')}")
    print(f"Message: {health.get('message', 'No message')}")
    print(f"Indexes built: {health.get('indexes_built', False)}")
    print(f"RAG ready: {health.get('rag_ready', False)}")
    print()

    if health.get("status") != "healthy":
        print("❌ API is not healthy. Please check the server.")
        return

    # Test queries
    test_queries = [
        "What is Optimus?",
        "Tell me about recommendation rejection reasons",
        "What are the best practices for Optimus?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Test Query {i}: {query}")
        print("-" * 40)

        # Test main query endpoint
        result = test_query(query)

        if "error" not in result:
            print(f"✅ Query processed in {result.get('processing_time_ms', 0):.1f}ms")
            print(f"📄 Text results: {len(result.get('text_results', []))}")
            print(f"📊 Table results: {len(result.get('table_results', []))}")
            print(f"🤖 Answer: {result.get('answer', 'No answer')[:100]}...")
        else:
            print(f"❌ Query failed: {result['error']}")

        print()

    # Test search endpoint
    print("🔎 Testing search endpoint...")
    search_results = test_search("What is Optimus?", text_k=2, table_k=1)

    if search_results:
        print(f"✅ Found {len(search_results)} search results")
        for i, result in enumerate(search_results[:3], 1):  # Show top 3
            if result.get("text"):
                print(f"  {i}. Text: {result['text'][:50]}... (score: {result['score']:.3f})")
            elif result.get("row_text"):
                print(f"  {i}. Table: {result['row_text'][:50]}... (score: {result['score']:.3f})")
    else:
        print("❌ No search results found")

    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    main()
