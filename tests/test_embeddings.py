"""Unit tests for embeddings module."""

import pytest
from src.retrieval.embeddings import EmbeddingGenerator, generate_embedding


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create an embedding generator for testing."""
        return EmbeddingGenerator()
    
    def test_embed_text(self, generator):
        """Test generating embedding for single text."""
        text = "Hello, world!"
        embedding = generator.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # text-embedding-3-small dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_empty_text(self, generator):
        """Test handling of empty text."""
        embedding = generator.embed_text("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(x == 0.0 for x in embedding)  # Zero vector
    
    def test_embed_batch(self, generator):
        """Test batched embedding generation."""
        texts = ["Hello", "World", "Test"]
        embeddings = generator.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
    
    def test_cosine_similarity(self, generator):
        """Test cosine similarity calculation."""
        # Use phrases instead of single words for better results
        emb1 = generator.embed_text("Dogs are loyal animals")
        emb2 = generator.embed_text("Cats are friendly pets")
        emb3 = generator.embed_text("Cars have four wheels")
        
        # Similar phrases should have higher similarity than dissimilar ones
        similarity_pets = generator.cosine_similarity(emb1, emb2)
        similarity_different = generator.cosine_similarity(emb1, emb3)
        
        # Pets should be more similar to each other than to vehicles
        assert similarity_pets > similarity_different
        
        # Both should be in reasonable range
        assert 0.0 <= similarity_pets <= 1.0
        assert 0.0 <= similarity_different <= 1.0
        
        # Pets should have at least moderate similarity
        assert similarity_pets > 0.45, f"Expected pet similarity > 0.45, got {similarity_pets}"
    
    def test_identical_texts_have_perfect_similarity(self, generator):
        """Test that identical texts have similarity of 1.0."""
        text = "Machine learning is powerful"
        emb1 = generator.embed_text(text)
        emb2 = generator.embed_text(text)
        
        similarity = generator.cosine_similarity(emb1, emb2)
        assert similarity > 0.99  # Should be very close to 1.0
    
    def test_similarity_is_symmetric(self, generator):
        """Test that similarity(A, B) == similarity(B, A)."""
        text1 = "Artificial intelligence"
        text2 = "Machine learning"
        
        emb1 = generator.embed_text(text1)
        emb2 = generator.embed_text(text2)
        
        sim_12 = generator.cosine_similarity(emb1, emb2)
        sim_21 = generator.cosine_similarity(emb2, emb1)
        
        assert abs(sim_12 - sim_21) < 0.0001  # Should be identical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])