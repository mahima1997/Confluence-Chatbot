#!/usr/bin/env python3
"""
rag_api.py

FastAPI web service for the Multimodal RAG system.
Provides REST endpoints for querying the RAG system.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import logging
from pathlib import Path

# Import our RAG system
from multimodal_rag import MultimodalRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="API for querying the Multimodal Retrieval-Augmented Generation system",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Global RAG instance (initialized lazily)
rag_instance: Optional[MultimodalRAG] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    top_k_text: Optional[int] = Field(5, ge=1, le=20, description="Number of text results to return")
    top_k_table: Optional[int] = Field(3, ge=1, le=20, description="Number of table results to return")
    include_context: Optional[bool] = Field(True, description="Whether to include full context in response")

class QueryResponse(BaseModel):
    query: str
    text_results: List[Dict[str, Any]]
    table_results: List[Dict[str, Any]]
    answer: str
    total_results: int
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    indexes_built: bool
    rag_ready: bool

class SearchResult(BaseModel):
    score: float
    text: Optional[str] = None
    row_text: Optional[str] = None
    source_file: str
    index: int
    table_idx: Optional[int] = None
    cells: Optional[List[str]] = None
    is_complete_table: Optional[bool] = None
    matched_by_headers: Optional[bool] = None
    table_headers: Optional[List[str]] = None

# Helper function to get or initialize RAG instance
def get_rag_instance() -> MultimodalRAG:
    """Get or initialize the RAG instance"""
    global rag_instance
    if rag_instance is None:
        logger.info("Initializing RAG system...")
        rag_instance = MultimodalRAG()

        # Check if indexes exist, build if needed
        base_dir = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot")
        text_index = base_dir / "rag_system" / "text_index.faiss"
        table_index = base_dir / "rag_system" / "table_index.faiss"
        metadata = base_dir / "rag_system" / "metadata.json"

        if not (text_index.exists() and table_index.exists() and metadata.exists()):
            logger.info("Building RAG indexes...")
            rag_instance.build_indexes()
        else:
            logger.info("Loading existing indexes...")
            rag_instance.load_indexes()

        logger.info("RAG system initialized successfully!")

    return rag_instance

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_rag_instance()

        # Check if indexes are available
        base_dir = Path("/Users/nishantgupta/Desktop/MahimuProject/Confluence-Chatbot")
        text_index = base_dir / "rag_system" / "text_index.faiss"
        table_index = base_dir / "rag_system" / "table_index.faiss"
        metadata = base_dir / "rag_system" / "metadata.json"

        indexes_built = text_index.exists() and table_index.exists() and metadata.exists()

        return HealthResponse(
            status="healthy",
            message="RAG system is operational",
            indexes_built=indexes_built,
            rag_ready=rag is not None
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            message=f"System error: {str(e)}",
            indexes_built=False,
            rag_ready=False
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main query endpoint for the RAG system"""
    import time

    start_time = time.time()

    try:
        # Get RAG instance
        rag = get_rag_instance()

        # Perform retrieval
        logger.info(f"Processing query: {request.query}")
        results = rag.retrieve(
            query=request.query,
            top_k_text=request.top_k_text,
            top_k_table=request.top_k_table
        )

        # Generate answer
        answer = rag.generate_answer(request.query, results)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Format response
        response_data = {
            "query": request.query,
            "text_results": [
                {
                    "score": result["score"],
                    "text": result["text"] if request.include_context else result["text"][:100] + "...",
                    "source_file": result["source_file"],
                    "index": result["index"]
                }
                for result in results["text"]
            ],
            "table_results": [
                {
                    "score": result["score"],
                    "row_text": result["row_text"] if request.include_context else result["row_text"][:100] + "...",
                    "source_file": result["source_file"],
                    "index": result["index"],
                    "table_idx": result.get("table_idx"),
                    "cells": result.get("cells"),
                    "is_complete_table": result.get("is_complete_table", False),
                    "matched_by_headers": result.get("matched_by_headers", False),
                    "table_headers": result.get("table_headers", [])
                }
                for result in results["table"]
            ],
            "answer": answer,
            "total_results": len(results["text"]) + len(results["table"]),
            "processing_time_ms": round(processing_time, 2)
        }

        logger.info(f"Query processed in {processing_time:.2f}ms with {response_data['total_results']} results")
        return response_data

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        processing_time = (time.time() - start_time) * 1000

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "query": request.query,
                "text_results": [],
                "table_results": [],
                "answer": f"Error processing query: {str(e)}",
                "total_results": 0,
                "processing_time_ms": round(processing_time, 2),
                "error": True
            }
        )

@app.get("/search", response_model=List[SearchResult])
async def search_endpoint(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    text_k: int = Query(5, ge=1, le=20, description="Number of text results"),
    table_k: int = Query(3, ge=1, le=20, description="Number of table results")
):
    """Simple search endpoint (GET request)"""
    try:
        rag = get_rag_instance()

        results = rag.retrieve(query=q, top_k_text=text_k, top_k_table=table_k)

        # Combine and sort results by score
        all_results = []

        for result in results["text"]:
            all_results.append(SearchResult(
                score=result["score"],
                text=result["text"],
                source_file=result["source_file"],
                index=result["index"]
            ))

        for result in results["table"]:
            all_results.append(SearchResult(
                score=result["score"],
                row_text=result["row_text"],
                source_file=result["source_file"],
                index=result["index"],
                table_idx=result.get("table_idx"),
                cells=result.get("cells"),
                is_complete_table=result.get("is_complete_table"),
                matched_by_headers=result.get("matched_by_headers"),
                table_headers=result.get("table_headers")
            ))

        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:text_k + table_k]  # Limit total results

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rebuild-indexes")
async def rebuild_indexes():
    """Endpoint to rebuild RAG indexes (useful if new data is added)"""
    try:
        global rag_instance
        logger.info("Rebuilding RAG indexes...")

        # Create new instance to force rebuild
        rag_instance = MultimodalRAG()
        rag_instance.build_indexes()

        return {"message": "Indexes rebuilt successfully", "status": "success"}

    except Exception as e:
        logger.error(f"Index rebuild failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multimodal RAG API",
        "version": "1.0.0",
        "description": "API for querying multimodal RAG system with text and table search",
        "endpoints": {
            "health": "GET /health - Health check",
            "query": "POST /query - Main query endpoint",
            "search": "GET /search - Simple search endpoint",
            "rebuild": "POST /rebuild-indexes - Rebuild indexes",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

if __name__ == "__main__":
    # Run the server
    logger.info("Starting Multimodal RAG API server...")
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
