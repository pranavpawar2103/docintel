"""
Unit tests for text chunker.
"""

import pytest
from src.ingestion.chunker import TextChunker, chunk_document, ChunkingStats
from src.ingestion.parser import DocumentChunk


class TestTextChunker:
    """Test suite for TextChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with small chunk size for testing."""
        return TextChunker(chunk_size=50, chunk_overlap=10)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience without being explicitly 
        programmed.
        
        Types of Machine Learning
        
        There are three main types: supervised learning, unsupervised learning, 
        and reinforcement learning. Each has its own use cases and algorithms.
        
        Applications
        
        Machine learning is used in recommendation systems, image recognition, 
        natural language processing, and many other domains.
        """
    
    def test_token_counting(self, chunker):
        """Test that token counting works correctly."""
        text = "Hello, world!"
        token_count = chunker.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < len(text)  # Tokens != characters
    
    def test_chunk_by_paragraphs(self, chunker):
        """Test paragraph-based chunking."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        paragraphs = chunker._chunk_by_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert "Paragraph one" in paragraphs[0]
        assert "Paragraph two" in paragraphs[1]
        assert "Paragraph three" in paragraphs[2]
    
    def test_chunk_by_sentences(self, chunker):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._chunk_by_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    def test_chunk_text_creates_document_chunks(self, chunker, sample_text):
        """Test that chunk_text returns DocumentChunk objects."""
        chunks = chunker.chunk_text(
            sample_text,
            document_name="test.txt",
            document_id="doc_001"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # Check first chunk has correct attributes
        first_chunk = chunks[0]
        assert first_chunk.document_name == "test.txt"
        assert first_chunk.document_id == "doc_001"
        assert first_chunk.chunk_index == 0
        assert len(first_chunk.text) > 0
    
    def test_chunk_size_respected(self, chunker, sample_text):
        """Test that chunks don't exceed max token limit."""
        chunks = chunker.chunk_text(
            sample_text,
            document_name="test.txt",
            document_id="doc_001"
        )
        
        for chunk in chunks:
            token_count = chunker.count_tokens(chunk.text)
            # Allow small margin for overlap
            assert token_count <= chunker.chunk_size + chunker.chunk_overlap
    
    def test_overlap_applied(self):
        """Test that overlap is added between chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        # Create text that will definitely produce multiple chunks
        text = " ".join([f"Word{i}" for i in range(100)])
        
        chunks = chunker.chunk_text(
            text,
            document_name="test.txt",
            document_id="doc_001"
        )
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            # (At least one word should appear in both)
            for i in range(len(chunks) - 1):
                chunk1_words = chunks[i].text.split()
                chunk2_words = chunks[i + 1].text.split()
                
                # At least some overlap
                overlap = set(chunk1_words) & set(chunk2_words)
                assert len(overlap) > 0, "Chunks should have overlap"
    
    def test_empty_text(self, chunker):
        """Test handling of empty text."""
        chunks = chunker.chunk_text(
            "",
            document_name="empty.txt",
            document_id="doc_001"
        )
        
        assert len(chunks) == 0
    
    def test_metadata_preserved(self, chunker):
        """Test that metadata is preserved in chunks."""
        metadata = {
            'author': 'Test Author',
            'num_pages': 10,
            'custom_field': 'custom_value'
        }
        
        chunks = chunker.chunk_text(
            "Test text for metadata preservation.",
            document_name="test.txt",
            document_id="doc_001",
            metadata=metadata
        )
        
        assert len(chunks) > 0
        first_chunk = chunks[0]
        
        assert first_chunk.metadata['author'] == 'Test Author'
        assert first_chunk.metadata['num_pages'] == 10
        assert first_chunk.metadata['custom_field'] == 'custom_value'
        assert 'chunk_tokens' in first_chunk.metadata
        assert 'chunk_chars' in first_chunk.metadata
    
    def test_convenience_function(self, sample_text):
        """Test the chunk_document convenience function."""
        chunks = chunk_document(
            sample_text,
            document_name="test.txt",
            document_id="doc_001"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)


class TestChunkingStats:
    """Test suite for ChunkingStats dataclass."""
    
    def test_stats_creation(self):
        """Test creating ChunkingStats."""
        stats = ChunkingStats(
            original_text_length=1000,
            num_chunks=10,
            avg_chunk_size=100,
            min_chunk_size=50,
            max_chunk_size=150,
            total_tokens=1000
        )
        
        assert stats.num_chunks == 10
        assert stats.avg_chunk_size == 100
        assert stats.total_tokens == 1000
    
    def test_stats_repr(self):
        """Test string representation of stats."""
        stats = ChunkingStats(
            original_text_length=1000,
            num_chunks=10,
            avg_chunk_size=100,
            min_chunk_size=50,
            max_chunk_size=150,
            total_tokens=1000
        )
        
        repr_str = repr(stats)
        assert "chunks=10" in repr_str
        assert "avg_size=100" in repr_str
        assert "total_tokens=1000" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    