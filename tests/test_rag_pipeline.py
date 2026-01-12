"""Unit tests for RAG pipeline."""

import pytest
from src.retrieval.rag_pipeline import RAGPipeline
from src.ingestion.pipeline import IngestionPipeline


class TestRAGPipeline:
    """Test suite for RAG pipeline."""
    
    @pytest.fixture
    def setup_test_data(self, tmp_path):
        """Set up test documents and RAG pipeline."""
        from src.retrieval.vectorstore import VectorStore
        
        # Create vector store in temp directory
        vector_store = VectorStore(
            collection_name="test_rag",
            persist_directory=str(tmp_path / "vectordb")
        )
        
        # Ingest test document
        ingestion = IngestionPipeline(vector_store=vector_store)
        ingestion.ingest_document("tests/test_documents/sample.txt")
        
        # Create RAG pipeline
        rag = RAGPipeline(vector_store=vector_store)
        
        yield rag
        
        # Cleanup
        vector_store.reset()
    
    def test_basic_query(self, setup_test_data):
        """Test basic question answering."""
        rag = setup_test_data
        
        response = rag.query("What is RAG?")
        
        assert isinstance(response, dict)
        assert 'answer' in response
        assert 'sources' in response
        assert 'confidence' in response
        assert len(response['answer']) > 0
    
    def test_query_with_no_results(self, setup_test_data):
        """Test query when no relevant context found."""
        rag = setup_test_data
        
        # Query about something not in documents
        response = rag.query("What is quantum computing?")
        
        assert isinstance(response, dict)
        assert 'answer' in response
        # Should indicate no information found
        assert 'couldn' in response['answer'].lower() or 'don\'t have' in response['answer'].lower()
    
    def test_sources_included(self, setup_test_data):
        """Test that sources are included in response."""
        rag = setup_test_data
        
        response = rag.query("What is retrieval?")
        
        assert 'sources' in response
        assert isinstance(response['sources'], list)
        if response['sources']:
            source = response['sources'][0]
            assert 'document_name' in source
            assert 'page_number' in source
    
    def test_conversation_history(self, setup_test_data):
        """Test conversation history tracking."""
        rag = setup_test_data
        
        # Ask multiple questions
        rag.query("What is RAG?")
        rag.query("How does retrieval work?")
        
        history = rag.get_conversation_history()
        
        assert len(history) >= 2
        assert all('question' in entry for entry in history)
        assert all('answer' in entry for entry in history)
    
    def test_statistics(self, setup_test_data):
        """Test pipeline statistics."""
        rag = setup_test_data
        
        stats = rag.get_statistics()
        
        assert 'total_documents' in stats
        assert 'total_chunks' in stats
        assert stats['total_documents'] > 0
        assert stats['total_chunks'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])