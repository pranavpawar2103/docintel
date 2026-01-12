"""
Text Chunking Module
Intelligently splits documents into optimal-sized chunks for embeddings and retrieval.

Key Concepts:
- Semantic boundaries: Split at paragraphs/sentences, not mid-word
- Token limits: Respect embedding model constraints (512 tokens)
- Overlap: Preserve context between chunks (50 tokens)
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
    Intelligent text chunker with multiple strategies.
    
    This is the heart of our RAG system. Good chunking = good retrieval.
    Bad chunking = garbage results, no matter how good your embeddings are.
    
    Strategy:
    1. Split by paragraphs (semantic boundaries)
    2. If paragraph > max_tokens, split by sentences
    3. If sentence > max_tokens, split by tokens (last resort)
    4. Add overlap between chunks
    5. Preserve metadata (page numbers, document info)
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"  # GPT-4, GPT-3.5-turbo encoding
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk (default from config)
            chunk_overlap: Token overlap between chunks (default from config)
            encoding_name: Tiktoken encoding to use
        
        Why these defaults:
        - 512 tokens: Good balance between context and precision
        - 50 token overlap: Preserves context at boundaries
        - cl100k_base: Works with modern OpenAI models
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize tokenizer
        # This is how we count tokens (same way OpenAI does)
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
        """
        Count tokens in text using tiktoken.
        
        Why we need this:
        - LLMs charge by token, not by character
        - "Hello" = 1 token, but "antidisestablishmentarianism" = 6 tokens
        - Need accurate counts to stay within model limits
        """
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
        
        This is the main entry point. It orchestrates all chunking strategies.
        
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
        
        # Strategy 1: Try to split by paragraphs first (most semantic)
        chunks = self._chunk_by_paragraphs(text)
        
        # Strategy 2: If any chunk is too large, split by sentences
        final_chunks = []
        for chunk_text in chunks:
            if self.count_tokens(chunk_text) > self.chunk_size:
                # Chunk is too large, split by sentences
                sentence_chunks = self._chunk_by_sentences(chunk_text)
                final_chunks.extend(sentence_chunks)
            else:
                final_chunks.append(chunk_text)
        
        # Strategy 3: If any chunk STILL too large, force split by tokens
        validated_chunks = []
        for chunk_text in final_chunks:
            if self.count_tokens(chunk_text) > self.chunk_size:
                # Last resort: force split by tokens
                token_chunks = self._chunk_by_tokens(chunk_text)
                validated_chunks.extend(token_chunks)
            else:
                validated_chunks.append(chunk_text)
        
        # Add overlap between chunks
        overlapped_chunks = self._add_overlap(validated_chunks)
        
        # Convert to DocumentChunk objects with metadata
        document_chunks = self._create_document_chunks(
            overlapped_chunks,
            document_name,
            document_id,
            metadata
        )
        
        # Log statistics
        stats = self._calculate_stats(text, document_chunks)
        logger.info(f"Chunking complete: {stats}")
        
        return document_chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text by paragraphs (double newlines).
        
        Why paragraphs:
        - Natural semantic boundaries
        - Usually contain complete thoughts
        - Preserve document structure
        
        Examples:
        "Intro paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        → ["Intro paragraph.", "Second paragraph.", "Third paragraph."]
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        logger.debug(f"Split text into {len(paragraphs)} paragraphs")
        return paragraphs
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences.
        
        Why sentences:
        - Fallback when paragraphs are too large
        - Still maintains grammatical completeness
        - Better than arbitrary character splits
        
        Regex explanation:
        - (?<=[.!?]) : Match after punctuation
        - \\s+ : Match whitespace
        - (?=[A-Z]) : Match before capital letter
        This ensures we split "Hello. World" but not "Mr. Smith"
        """
        # Split on sentence boundaries (. ! ? followed by space and capital letter)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.debug(f"Split text into {len(sentences)} sentences")
        return sentences
    
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """
        Force split text by token count (last resort).
        
        Why this is last resort:
        - Loses semantic meaning
        - Can split mid-sentence
        - Only used when paragraph AND sentence splitting failed
        
        Algorithm:
        1. Encode text to tokens
        2. Split tokens into chunks of chunk_size
        3. Decode back to text
        """
        # Encode to tokens
        tokens = self.encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        logger.debug(f"Force split into {len(chunks)} token-based chunks")
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Why overlap is critical:
        - Prevents information loss at boundaries
        - Improves retrieval for queries spanning chunk boundaries
        
        Example without overlap:
        Chunk 1: "...the key benefit is accuracy."
        Chunk 2: "Another advantage is speed..."
        Query: "What are the benefits?" → Might only retrieve Chunk 1
        
        Example with overlap:
        Chunk 1: "...the key benefit is accuracy. Another advantage"
        Chunk 2: "accuracy. Another advantage is speed..."
        Query: "What are the benefits?" → Retrieves both chunks!
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix overlap needed
                overlapped.append(chunk)
            else:
                # Get tokens from previous chunk for overlap
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
    
    def _create_document_chunks(
        self,
        chunks: List[str],
        document_name: str,
        document_id: str,
        metadata: dict
    ) -> List[DocumentChunk]:
        """
        Convert raw text chunks to DocumentChunk objects with metadata.
        
        This preserves all the important information:
        - Source document
        - Position in document (chunk_index)
        - Page numbers (if available from PDF)
        - Custom metadata
        """
        document_chunks = []
        
        for idx, chunk_text in enumerate(chunks):
            # Determine page number (if available from metadata)
            page_number = self._estimate_page_number(
                idx,
                len(chunks),
                metadata.get('num_pages')
            )
            
            chunk = DocumentChunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=idx,
                document_name=document_name,
                document_id=document_id,
                metadata={
                    **metadata,  # Preserve original metadata
                    'chunk_tokens': self.count_tokens(chunk_text),
                    'chunk_chars': len(chunk_text),
                }
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def _estimate_page_number(
        self,
        chunk_index: int,
        total_chunks: int,
        num_pages: Optional[int]
    ) -> Optional[int]:
        """
        Estimate which page a chunk comes from.
        
        Why estimate:
        - For PDFs, we know total pages
        - We can approximate: chunk 10 of 100 total chunks ≈ page 10 of 100 pages
        
        Limitation:
        - Assumes even distribution (not always true)
        - Better: track page during parsing (we'll improve this later)
        """
        if num_pages is None:
            return None
        
        # Simple estimation: distribute chunks evenly across pages
        estimated_page = int((chunk_index / total_chunks) * num_pages) + 1
        return min(estimated_page, num_pages)  # Cap at max page
    
    def _calculate_stats(
        self,
        original_text: str,
        chunks: List[DocumentChunk]
    ) -> ChunkingStats:
        """Calculate statistics about the chunking process."""
        chunk_sizes = [self.count_tokens(chunk.text) for chunk in chunks]
        
        return ChunkingStats(
            original_text_length=len(original_text),
            num_chunks=len(chunks),
            avg_chunk_size=sum(chunk_sizes) // len(chunk_sizes) if chunks else 0,
            min_chunk_size=min(chunk_sizes) if chunks else 0,
            max_chunk_size=max(chunk_sizes) if chunks else 0,
            total_tokens=sum(chunk_sizes)
        )


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