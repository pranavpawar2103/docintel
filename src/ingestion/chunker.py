"""
Text Chunking Module
Intelligently splits documents into optimal-sized chunks for embeddings and retrieval.

Key Concepts:
- Token-based chunking: Respect embedding model constraints
- Sentence boundaries: Don't break mid-sentence
- Overlap: Preserve context between chunks
- Metadata preservation: Track page numbers and positions
"""

from typing import List, Optional
import re
import logging
import tiktoken
from dataclasses import dataclass

from src.ingestion.parser import DocumentChunk
from src.utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStats:
    """Statistics about the chunking process."""
    original_text_length: int
    num_chunks: int
    avg_chunk_size: int
    min_chunk_size: int
    max_chunk_size: int
    total_tokens: int
    
    def __repr__(self) -> str:
        return (f"ChunkingStats(chunks={self.num_chunks}, "
                f"avg_size={self.avg_chunk_size}, "
                f"total_tokens={self.total_tokens})")


class TextChunker:
    """
    Intelligent text chunker optimized for RAG systems.
    
    Fixed version that creates properly-sized chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk (default from config)
            chunk_overlap: Token overlap between chunks (default from config)
            encoding_name: Tiktoken encoding to use
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}, using default")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(
            f"Initialized TextChunker: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, encoding={encoding_name}"
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def chunk_text(
        self,
        text: str,
        document_name: str,
        document_id: str,
        metadata: dict = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into optimal-sized pieces.
        
        FIXED VERSION - creates proper-sized chunks.
        
        Args:
            text: The full text to chunk
            document_name: Name of source document
            document_id: Unique identifier
            metadata: Additional metadata to preserve
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        metadata = metadata or {}
        
        # For PDFs with page information, use page-aware chunking
        if 'pages_text' in metadata and metadata['pages_text']:
            return self._chunk_pdf_with_pages(
                metadata['pages_text'],
                document_name,
                document_id,
                metadata
            )
        
        # For plain text, use sentence-aware token chunking
        chunks = self._chunk_by_sentences(text)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                text=chunk_text,
                page_number=None,
                chunk_index=idx,
                document_name=document_name,
                document_id=document_id,
                metadata={
                    **metadata,
                    'chunk_tokens': self.count_tokens(chunk_text),
                    'chunk_chars': len(chunk_text),
                }
            )
            document_chunks.append(chunk)
        
        # Log statistics
        stats = self._calculate_stats_from_chunks(document_chunks)
        logger.info(f"Chunking complete: {stats}")
        
        return document_chunks
    
    def _chunk_pdf_with_pages(
        self,
        pages_text: List[str],
        document_name: str,
        document_id: str,
        metadata: dict
    ) -> List[DocumentChunk]:
        """
        Chunk PDF intelligently with page tracking.
        
        Strategy:
        - Accumulate text from pages until we reach chunk_size
        - Split at that point using sentence boundaries
        - Track which page each chunk starts on
        """
        all_chunks = []
        
        # Combine all pages into one text with page markers
        combined_text = ""
        page_boundaries = [0]  # Character positions where pages start
        
        for page_text in pages_text:
            combined_text += page_text + "\n\n"
            page_boundaries.append(len(combined_text))
        
        # Chunk the combined text
        text_chunks = self._chunk_by_sentences(combined_text)
        text_chunks = self._add_overlap(text_chunks)
        
        # Determine page number for each chunk
        current_pos = 0
        for idx, chunk_text in enumerate(text_chunks):
            # Find which page this chunk starts on
            page_num = 1
            for p_idx, boundary in enumerate(page_boundaries[1:], start=1):
                if current_pos < boundary:
                    page_num = p_idx
                    break
            
            chunk = DocumentChunk(
                text=chunk_text,
                page_number=page_num,
                chunk_index=idx,
                document_name=document_name,
                document_id=document_id,
                metadata={
                    **metadata,
                    'chunk_tokens': self.count_tokens(chunk_text),
                    'chunk_chars': len(chunk_text),
                }
            )
            all_chunks.append(chunk)
            
            # Move position forward (accounting for overlap)
            current_pos += len(chunk_text) - (len(chunk_text) // 4)  # Rough estimate
        
        stats = self._calculate_stats_from_chunks(all_chunks)
        logger.info(f"Chunking complete: {stats}")
        
        return all_chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Simple, robust token-based chunking.
        Splits at paragraph breaks when possible.
        """
        # Clean the text first - remove excessive newlines
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines → single
        text = re.sub(r'\n', ' ', text)    # Newlines → spaces
        text = re.sub(r'\s+', ' ', text)   # Multiple spaces → single
        
        # Split into sentences using simple regex
        # Match: . ! ? followed by space and capital letter OR end of string
        sentences = re.split(r'(?<=[.!?])(?:\s+(?=[A-Z])|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Build chunks by accumulating sentences
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens = self.count_tokens(current_chunk)
            
            # Handle oversized single sentences
            if sentence_tokens > self.chunk_size:
                if current_chunk and current_chunk != sentence:
                    chunks.append(current_chunk.strip())
                
                # Force split the long sentence
                tokens = self.encoding.encode(sentence)
                for i in range(0, len(tokens), self.chunk_size):
                    chunk_tokens = tokens[i:i + self.chunk_size]
                    chunks.append(self.encoding.decode(chunk_tokens))
                
                current_chunk = ""
                current_tokens = 0
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _force_split_by_tokens(self, text: str) -> List[str]:
        """
        Force split text by tokens (when a single sentence is too large).
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Overlap helps preserve context at boundaries.
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix overlap
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                prev_tokens = self.encoding.encode(prev_chunk)
                
                # Take last N tokens from previous chunk
                overlap_tokens = prev_tokens[-self.chunk_overlap:]
                overlap_text = self.encoding.decode(overlap_tokens)
                
                # Prepend overlap to current chunk
                overlapped_chunk = overlap_text + " " + chunk
                overlapped.append(overlapped_chunk)
        
        logger.debug(f"Added overlap to {len(chunks)} chunks")
        return overlapped
    
    def _calculate_stats_from_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> ChunkingStats:
        """Calculate statistics from DocumentChunk objects."""
        if not chunks:
            return ChunkingStats(0, 0, 0, 0, 0, 0)
        
        chunk_sizes = [self.count_tokens(chunk.text) for chunk in chunks]
        total_chars = sum(len(chunk.text) for chunk in chunks)
        
        return ChunkingStats(
            original_text_length=total_chars,
            num_chunks=len(chunks),
            avg_chunk_size=sum(chunk_sizes) // len(chunk_sizes) if chunks else 0,
            min_chunk_size=min(chunk_sizes) if chunk_sizes else 0,
            max_chunk_size=max(chunk_sizes) if chunk_sizes else 0,
            total_tokens=sum(chunk_sizes)
        )
    
    def _calculate_stats(
        self,
        original_text: str,
        chunks: List[DocumentChunk]
    ) -> ChunkingStats:
        """Calculate statistics about the chunking process."""
        return self._calculate_stats_from_chunks(chunks)


# Convenience function
def chunk_document(
    text: str,
    document_name: str,
    document_id: str,
    metadata: dict = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[DocumentChunk]:
    """
    Convenience function to chunk a document.
    
    Usage:
        chunks = chunk_document(text, "paper.pdf", "doc_123")
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text, document_name, document_id, metadata)