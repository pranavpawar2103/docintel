"""
FastAPI Backend for DocIntel RAG System

This provides REST API endpoints for:
- Document upload and management
- Question answering
- System statistics

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
from pathlib import Path
import shutil
from datetime import datetime
import time
from typing import Optional, Generator
import json

from src.api.models import (
    QueryRequest, QueryResponse, DocumentUploadResponse,
    DocumentListResponse, StatisticsResponse, ErrorResponse, HealthResponse
)
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="DocIntel API",
    description="Intelligent Document Analysis System with RAG",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc at http://localhost:8000/redoc
)

# Add CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State (initialized on startup)
# ============================================================================

ingestion_pipeline: Optional[IngestionPipeline] = None
rag_pipeline: Optional[RAGPipeline] = None
app_start_time: float = 0


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize pipelines on app startup."""
    global ingestion_pipeline, rag_pipeline, app_start_time
    
    logger.info("🚀 Starting DocIntel API...")
    app_start_time = time.time()
    
    try:
        # Initialize pipelines
        ingestion_pipeline = IngestionPipeline()
        rag_pipeline = RAGPipeline(vector_store=ingestion_pipeline.vector_store)
        
        logger.info("✅ Pipelines initialized successfully")
        logger.info(f"   Documents: {rag_pipeline.get_statistics()['total_documents']}")
        logger.info(f"   Chunks: {rag_pipeline.get_statistics()['total_chunks']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipelines: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down DocIntel API...")


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and version.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"]
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and index a document.
    
    Supported formats: PDF, DOCX, TXT, MD
    
    Process:
    1. Save uploaded file
    2. Parse and chunk document
    3. Generate embeddings
    4. Store in vector database
    
    Returns document metadata and ingestion statistics.
    """
    logger.info(f"Received upload request: {file.filename}")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {allowed_extensions}"
        )
    
    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        
        # Ingest document
        result = ingestion_pipeline.ingest_document(str(file_path))
        
        if result['success']:
            logger.info(f"✅ Document ingested: {result['document_id']}")
            
            return DocumentUploadResponse(
                success=True,
                document_id=result['document_id'],
                document_name=result['document_name'],
                num_chunks=result['num_chunks'],
                message=f"Successfully processed {result['num_chunks']} chunks"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        
        # Clean up file if ingestion failed
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
    finally:
        await file.close()


@app.get(
    "/api/documents",
    response_model=DocumentListResponse,
    tags=["Documents"]
)
async def list_documents():
    """
    List all indexed documents.
    
    Returns document IDs and statistics.
    """
    try:
        stats = rag_pipeline.get_statistics()
        
        return DocumentListResponse(
            documents=stats['documents'],
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks']
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/documents/{document_id}",
    tags=["Documents"]
)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: ID of document to delete
        
    Returns success message with number of chunks deleted.
    """
    try:
        num_deleted = rag_pipeline.vector_store.delete_document(document_id)
        
        if num_deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        logger.info(f"✅ Deleted document {document_id} ({num_deleted} chunks)")
        
        return {
            "success": True,
            "message": f"Deleted {num_deleted} chunks from document {document_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Query Endpoints
# ============================================================================

@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["Query"]
)
async def query_documents(request: QueryRequest):
    """
    Ask a question and get an answer with sources.
    
    Process:
    1. Embed query
    2. Search vector database
    3. Generate answer with LLM
    4. Return answer with citations
    
    Example request:
```json
    {
        "question": "What is machine learning?",
        "n_results": 5,
        "include_context": false
    }
```
    """
    logger.info(f"Query received: '{request.question}'")
    
    try:
        # Query RAG pipeline
        response = rag_pipeline.query(
            question=request.question,
            n_results=request.n_results,
            filters=request.filters,
            include_context=request.include_context
        )
        
        # Convert to API response model
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.post(
    "/api/query/stream",
    tags=["Query"]
)
async def query_documents_stream(request: QueryRequest):
    """
    Ask a question and stream the answer word-by-word.
    
    Great for interactive UIs - users see response as it's generated.
    
    Returns Server-Sent Events (SSE) stream.
    """
    logger.info(f"Streaming query received: '{request.question}'")
    
    async def generate() -> Generator[str, None, None]:
        """Generate streaming response."""
        try:
            for chunk in rag_pipeline.query_stream(
                question=request.question,
                n_results=request.n_results,
                filters=request.filters
            ):
                # Format as Server-Sent Event
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Streaming query failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ============================================================================
# Statistics Endpoint
# ============================================================================

@app.get(
    "/api/stats",
    response_model=StatisticsResponse,
    tags=["System"]
)
async def get_statistics():
    """
    Get system statistics.
    
    Returns:
    - Number of documents
    - Number of chunks
    - Number of queries processed
    - Average confidence
    - Uptime
    """
    try:
        stats = rag_pipeline.get_statistics()
        uptime = time.time() - app_start_time
        
        return StatisticsResponse(
            **stats,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "DocIntel API",
        "version": "1.0.0",
        "description": "Intelligent Document Analysis System with RAG",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {str(exc)}")
    
    return ErrorResponse(
        error="Internal server error",
        detail=str(exc),
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# Run Server (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )