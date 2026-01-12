"""Unit tests for vector store."""

import pytest
from src.retrieval.vectorstore import VectorStore
from src.ingestion.parser import DocumentChunk


class TestVectorStore:
    """Test suite for VectorStore."""
    
    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a temporary vector store for testing."""
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=str(tmp_path / "vectordb")
        )
        # Clean up after each test
        yield store
        store.reset()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks."""
        return [
            DocumentChunk(
                text="Machine learning is a subset of artificial intelligence.",
                page_number=1,
                chunk_index=0,
                document_name="ml_basics.pdf",
                document_id="doc_001",
                metadata={'topic': 'ML'}
            ),
            DocumentChunk(
                text="Deep learning uses neural networks with multiple layers.",
                page_number=2,
                chunk_index=1,
                document_name="ml_basics.pdf",
                document_id="doc_001",
                metadata={'topic': 'DL'}
            ),
            DocumentChunk(
                text="Python is a popular programming language.",
                page_number=1,
                chunk_index=0,
                document_name="programming.pdf",
                document_id="doc_002",
                metadata={'topic': 'Programming'}
            )
        ]
    
    def test_add_documents(self, vector_store, sample_chunks):
        """Test adding documents to the store."""
        count = vector_store.add_documents(sample_chunks)
        
        assert count == 3
        assert vector_store.count_documents() == 3
    
    def test_search(self, vector_store, sample_chunks):
        """Test semantic search."""
        # Add documents
        vector_store.add_documents(sample_chunks)
        
        # Search for ML-related content
        results = vector_store.search("What is machine learning?", n_results=2)
        
        assert len(results) == 2
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)
        assert all('metadata' in r for r in results)
        
        # First result should be most relevant
        assert "machine learning" in results[0]['text'].lower()
    
    def test_delete_document(self, vector_store, sample_chunks):
        """Test deleting a document."""
        # Add documents
        vector_store.add_documents(sample_chunks)
        assert vector_store.count_documents() == 3
        
        # Delete one document
        deleted_count = vector_store.delete_document("doc_001")
        assert deleted_count == 2  # doc_001 has 2 chunks
        assert vector_store.count_documents() == 1  # Only doc_002 remains
    
    def test_list_documents(self, vector_store, sample_chunks):
        """Test listing documents."""
        vector_store.add_documents(sample_chunks)
        
        doc_ids = vector_store.list_documents()
        
        assert len(doc_ids) == 2  # doc_001 and doc_002
        assert "doc_001" in doc_ids
        assert "doc_002" in doc_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])