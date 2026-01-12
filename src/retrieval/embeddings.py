"""
Embeddings Module
Converts text into vector embeddings using OpenAI's embedding models.

Key Concepts:
- Embeddings capture semantic meaning of text
- Similar text → similar vectors (high cosine similarity)
- Enables semantic search (search by meaning, not keywords)
"""

from typing import List, Optional
import logging
import time
from openai import OpenAI
import numpy as np

from src.utils.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text using OpenAI's embedding models.
    
    Why OpenAI embeddings:
    - High quality (SOTA performance on retrieval benchmarks)
    - Fast (1000 texts in ~2 seconds)
    - Cost-effective ($0.02 per 1M tokens)
    - Easy to use (just one API call)
    
    Model: text-embedding-3-small
    - Dimensions: 1536
    - Max tokens: 8192
    - Cost: $0.02 / 1M tokens
    - Performance: Excellent for most use cases
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: Embedding model name (default from config)
            api_key: OpenAI API key (default from config)
            batch_size: Number of texts to embed in one API call
        
        Why batch_size matters:
        - API allows up to 2048 inputs per call
        - Batching reduces latency (fewer API calls)
        - We use 100 as safe default (balances speed and memory)
        """
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.openai_api_key
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.embed_text("Hello world")
            >>> len(embedding)
            1536
            >>> type(embedding[0])
            <class 'float'>
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector for empty text
            return [0.0] * 1536
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"  # Return as list of floats
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency).
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress
            
        Returns:
            List of embeddings (one per text)
            
        Why batching:
        - Single API call for multiple texts
        - 10x faster than calling embed_text() in a loop
        - Reduces API costs (fewer requests)
        
        Example:
            >>> texts = ["Hello", "World", "AI is cool"]
            >>> embeddings = generator.embed_batch(texts)
            >>> len(embeddings)
            3
            >>> len(embeddings[0])
            1536
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        # Filter out empty texts but preserve indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning("All texts are empty")
            return [[0.0] * 1536] * len(texts)
        
        all_embeddings = []
        total_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts in {total_batches} batches")
        
        # Process in batches
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                # Call API with batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                # Extract embeddings (preserve order)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting: small delay between batches
                if i + self.batch_size < len(valid_texts):
                    time.sleep(0.1)  # 100ms delay
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {str(e)}")
                raise
        
        # Reconstruct full list with zero vectors for empty texts
        final_embeddings = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                final_embeddings.append(all_embeddings[valid_idx])
                valid_idx += 1
            else:
                final_embeddings.append([0.0] * 1536)
        
        logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
        return final_embeddings
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity ranges from -1 to 1:
        - 1.0: Identical meaning
        - 0.0: No relationship
        - -1.0: Opposite meaning (rare in practice)
        
        Formula:
        similarity = (A · B) / (||A|| * ||B||)
        
        Where:
        - A · B = dot product
        - ||A|| = magnitude of vector A
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1)
            
        Example:
            >>> emb1 = generator.embed_text("dog")
            >>> emb2 = generator.embed_text("cat")
            >>> similarity = generator.cosine_similarity(emb1, emb2)
            >>> similarity
            0.85  # High similarity!
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        similarity = dot_product / (magnitude_a * magnitude_b)
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.
        
        Returns:
            Number of dimensions (1536 for text-embedding-3-small)
        """
        return 1536  # text-embedding-3-small dimension


# Convenience functions
def generate_embedding(text: str) -> List[float]:
    """
    Quick helper to generate a single embedding.
    
    Usage:
        embedding = generate_embedding("Hello world")
    """
    generator = EmbeddingGenerator()
    return generator.embed_text(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Quick helper to generate multiple embeddings.
    
    Usage:
        embeddings = generate_embeddings(["Hello", "World"])
    """
    generator = EmbeddingGenerator()
    return generator.embed_batch(texts)