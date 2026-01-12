"""
Unit tests for document parser.
"""

import pytest
from pathlib import Path
from src.ingestion.parser import DocumentParser, parse_document, DocumentChunk


class TestDocumentParser:
    """Test suite for DocumentParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return DocumentParser()
    
    @pytest.fixture
    def test_docs_dir(self):
        """Path to test documents directory."""
        return Path("tests/test_documents")
    
    def test_supported_extensions(self):
        """Test that supported extensions are defined correctly."""
        parser = DocumentParser()
        assert '.pdf' in parser.SUPPORTED_EXTENSIONS
        assert '.docx' in parser.SUPPORTED_EXTENSIONS
        assert '.txt' in parser.SUPPORTED_EXTENSIONS
        assert '.md' in parser.SUPPORTED_EXTENSIONS
    
    def test_parse_text_file(self, parser, test_docs_dir):
        """Test parsing a plain text file."""
        text_file = test_docs_dir / "sample.txt"
        
        # Create test file if it doesn't exist
        if not text_file.exists():
            test_docs_dir.mkdir(parents=True, exist_ok=True)
            text_file.write_text(
                "Introduction to RAG Systems\n\n"
                "Retrieval-Augmented Generation (RAG) is a technique..."
            )
        
        text, metadata = parser.parse(str(text_file))
        
        # Assertions
        assert isinstance(text, str)
        assert len(text) > 0
        assert "RAG" in text or "Retrieval" in text
        
        assert metadata['filename'] == 'sample.txt'
        assert metadata['num_lines'] > 0
        assert metadata['file_size'] > 0
    
    def test_parse_markdown_file(self, parser, test_docs_dir):
        """Test parsing a markdown file."""
        md_file = test_docs_dir / "sample.md"
        
        # Create test file if it doesn't exist
        if not md_file.exists():
            test_docs_dir.mkdir(parents=True, exist_ok=True)
            md_file.write_text(
                "# Machine Learning\n\n"
                "## Supervised Learning\n\n"
                "Supervised learning uses labeled data."
            )
        
        text, metadata = parser.parse(str(md_file))
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Machine Learning" in text or "Supervised" in text
        assert metadata['filename'] == 'sample.md'
    
    def test_parse_nonexistent_file(self, parser):
        """Test that parsing a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent_file.pdf")
    
    def test_parse_unsupported_format(self, parser, tmp_path):
        """Test that unsupported file formats raise ValueError."""
        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(str(unsupported_file))
    
    def test_convenience_function(self, test_docs_dir):
        """Test the parse_document convenience function."""
        text_file = test_docs_dir / "sample.txt"
        
        if not text_file.exists():
            test_docs_dir.mkdir(parents=True, exist_ok=True)
            text_file.write_text("Test content")
        
        text, metadata = parse_document(str(text_file))
        
        assert isinstance(text, str)
        assert isinstance(metadata, dict)
        assert len(text) > 0


class TestDocumentChunk:
    """Test suite for DocumentChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            text="This is a test chunk.",
            page_number=1,
            chunk_index=0,
            document_name="test.pdf",
            document_id="doc_123",
            metadata={'key': 'value'}
        )
        
        assert chunk.text == "This is a test chunk."
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0
        assert chunk.document_name == "test.pdf"
        assert chunk.document_id == "doc_123"
        assert chunk.metadata == {'key': 'value'}
    
    def test_chunk_repr(self):
        """Test string representation of DocumentChunk."""
        chunk = DocumentChunk(
            text="Short text",
            page_number=5,
            chunk_index=2,
            document_name="doc.pdf",
            document_id="doc_123",
            metadata={}
        )
        
        repr_str = repr(chunk)
        assert "doc.pdf" in repr_str
        assert "page=5" in repr_str
        assert "chunk=2" in repr_str
    
    def test_chunk_with_long_text(self):
        """Test repr with long text (should be truncated)."""
        long_text = "A" * 100
        chunk = DocumentChunk(
            text=long_text,
            page_number=1,
            chunk_index=0,
            document_name="test.pdf",
            document_id="doc_123",
            metadata={}
        )
        
        repr_str = repr(chunk)
        # Should be truncated to 50 chars + "..."
        assert len(chunk.text) == 100
        assert "..." in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])