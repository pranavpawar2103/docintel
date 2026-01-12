"""
API Models for Request/Response Validation
Uses Pydantic for automatic validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Request Models (What users send to API)
# ============================================================================

class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.
    
    Example:
        {
            "question": "What is machine learning?",
            "n_results": 5,
            "include_context": false
        }
    """
    question: str = Field(
        ...,
        description="The question to ask",
        min_length=1,
        max_length=1000,
        examples=["What is machine learning?"]
    )
    n_results: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve (uses config default if not specified)",
        ge=1,
        le=20
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for retrieval"
    )
    include_context: bool = Field(
        default=False,
        description="Whether to include retrieved chunks in response"
    )


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    document_id: str
    document_name: str
    num_chunks: int
    message: str


# ============================================================================
# Response Models (What API sends back to users)
# ============================================================================

class Source(BaseModel):
    """Source citation for an answer."""
    document_name: str = Field(description="Name of source document")
    page_number: Any = Field(description="Page number (may be 'N/A' for non-PDF)")
    relevance_score: float = Field(
        description="Relevance score (0-1)",
        ge=0.0,
        le=1.0
    )


class QueryResponse(BaseModel):
    """
    Response model for queries.
    
    Example:
        {
            "answer": "Machine learning is...",
            "sources": [...],
            "confidence": 0.85,
            "processing_time_ms": 2340
        }
    """
    answer: str = Field(description="Generated answer")
    sources: List[Source] = Field(
        default=[],
        description="Source citations"
    )
    confidence: float = Field(
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0
    )
    model: str = Field(description="LLM model used")
    tokens_used: int = Field(description="Total tokens consumed")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    timestamp: str = Field(description="ISO timestamp")
    retrieved_chunks: Optional[List[Dict]] = Field(
        default=None,
        description="Retrieved context chunks (if requested)"
    )


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    document_id: str
    document_name: str
    num_chunks: int
    upload_timestamp: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[str] = Field(description="List of document IDs")
    total_documents: int = Field(description="Total number of documents")
    total_chunks: int = Field(description="Total number of chunks")


class StatisticsResponse(BaseModel):
    """Response model for system statistics."""
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_confidence: float
    documents: List[str]
    uptime_seconds: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: str = Field(description="ISO timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Status (healthy/unhealthy)")
    version: str = Field(description="API version")
    timestamp: str = Field(description="ISO timestamp")