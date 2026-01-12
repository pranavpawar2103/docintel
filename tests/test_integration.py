"""
Integration tests for complete RAG pipeline.
Tests the entire flow: upload → process → query → response
"""

import pytest
import time
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.rag_pipeline import RAGPipeline
from src.retrieval.vectorstore import VectorStore


@pytest.mark.integration
class TestCompleteRAGFlow:
    """Integration tests for complete RAG workflow."""
    
    @pytest.fixture
    def setup_system(self, tmp_path):
        """Set up complete RAG system with test data."""
        # Create vector store in temp directory
        vector_store = VectorStore(
            collection_name="test_integration",
            persist_directory=str(tmp_path / "vectordb")
        )
        
        # Initialize pipelines
        ingestion = IngestionPipeline(vector_store=vector_store)
        rag = RAGPipeline(vector_store=vector_store)
        
        # Ingest test document
        test_doc = "tests/test_documents/sample.txt"
        if Path(test_doc).exists():
            ingestion.ingest_document(test_doc)
        
        yield {
            'ingestion': ingestion,
            'rag': rag,
            'vector_store': vector_store
        }
        
        # Cleanup
        vector_store.reset()
    
    def test_end_to_end_query_flow(self, setup_system):
        """Test complete query flow with timing."""
        rag = setup_system['rag']
        
        # Measure query time
        start_time = time.time()
        response = rag.query("What is RAG?")
        query_time = time.time() - start_time
        
        # Assertions
        assert 'answer' in response
        assert len(response['answer']) > 0
        assert response['confidence'] > 0.0
        assert 'sources' in response
        assert query_time < 5.0  # Should complete in <5 seconds
        
        # Verify metadata
        assert 'model' in response
        assert 'tokens_used' in response
        assert response['tokens_used'] > 0
    
    def test_query_with_no_relevant_context(self, setup_system):
        """Test query when no relevant documents exist."""
        rag = setup_system['rag']
        
        # Query about something definitely not in documents
        response = rag.query("What is quantum entanglement?")
        
        # Should handle gracefully
        assert 'answer' in response
        assert response['confidence'] == 0.0
        assert len(response['sources']) == 0
    
    def test_multiple_queries_performance(self, setup_system):
        """Test performance with multiple queries."""
        rag = setup_system['rag']
        
        queries = [
            "What is RAG?",
            "How does retrieval work?",
            "What are the benefits?",
        ]
        
        total_time = 0
        for query in queries:
            start = time.time()
            response = rag.query(query)
            total_time += time.time() - start
            
            assert 'answer' in response
        
        avg_time = total_time / len(queries)
        assert avg_time < 5.0  # Average should be <5s per query
    
    def test_document_deletion_and_requery(self, setup_system):
        """Test that deleting documents affects query results."""
        rag = setup_system['rag']
        vector_store = setup_system['vector_store']
        
        # Query before deletion
        response_before = rag.query("What is RAG?")
        
        # Delete all documents
        doc_ids = vector_store.list_documents()
        for doc_id in doc_ids:
            vector_store.delete_document(doc_id)
        
        # Query after deletion
        response_after = rag.query("What is RAG?")
        
        # Should have lower confidence after deletion
        assert response_after['confidence'] < response_before['confidence']
    
    def test_conversation_history_tracking(self, setup_system):
        """Test conversation history is tracked."""
        rag = setup_system['rag']
        
        # Ask multiple questions
        rag.query("What is RAG?")
        rag.query("How does it work?")
        
        history = rag.get_conversation_history()
        
        assert len(history) >= 2
        assert all('question' in entry for entry in history)
        assert all('answer' in entry for entry in history)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_ingestion_speed(self, tmp_path):
        """Benchmark document ingestion speed."""
        vector_store = VectorStore(
            collection_name="test_perf",
            persist_directory=str(tmp_path / "vectordb")
        )
        ingestion = IngestionPipeline(vector_store=vector_store)
        
        test_doc = "tests/test_documents/sample.txt"
        if not Path(test_doc).exists():
            pytest.skip("Test document not found")
        
        # Measure ingestion time
        start_time = time.time()
        result = ingestion.ingest_document(test_doc)
        ingestion_time = time.time() - start_time
        
        print(f"\n📊 Ingestion Performance:")
        print(f"   Time: {ingestion_time:.2f}s")
        print(f"   Chunks: {result['num_chunks']}")
        print(f"   Speed: {result['num_chunks']/ingestion_time:.1f} chunks/sec")
        
        # Cleanup
        vector_store.reset()
    
    def test_query_latency_distribution(self, tmp_path):
        """Measure query latency distribution."""
        vector_store = VectorStore(
            collection_name="test_latency",
            persist_directory=str(tmp_path / "vectordb")
        )
        ingestion = IngestionPipeline(vector_store=vector_store)
        
        test_doc = "tests/test_documents/sample.txt"
        if Path(test_doc).exists():
            ingestion.ingest_document(test_doc)
        
        rag = RAGPipeline(vector_store=vector_store)
        
        # Run multiple queries
        latencies = []
        queries = [
            "What is RAG?",
            "How does retrieval work?",
            "What are the benefits?",
            "Explain the system",
        ] * 3  # Run each 3 times
        
        for query in queries:
            start = time.time()
            rag.query(query)
            latencies.append(time.time() - start)
        
        # Calculate statistics
        import statistics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\n📊 Query Latency Distribution:")
        print(f"   Average: {avg_latency:.2f}s")
        print(f"   P50 (median): {p50_latency:.2f}s")
        print(f"   P95: {p95_latency:.2f}s")
        print(f"   Min: {min(latencies):.2f}s")
        print(f"   Max: {max(latencies):.2f}s")
        
        # Assertions
        assert avg_latency < 5.0  # Average should be reasonable
        assert p95_latency < 10.0  # P95 should be acceptable
        
        # Cleanup
        vector_store.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])