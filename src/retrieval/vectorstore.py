"""
Vector Store Module
Manages storage and retrieval of embeddings using ChromaDB.

Key Concepts:
- Vector database stores embeddings + metadata
- Fast similarity search using HNSW index
- Persistent storage to disk
"""

from typing import List, Dict, Optional, Tuple
import logging
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.parser import DocumentChunk
from src.retrieval.embeddings import EmbeddingGenerator
from src.utils.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for storing and searching document embeddings.
    
    Why ChromaDB:
    - Embedded (no server needed)
    - Persistent (saves to disk)
    - Fast (HNSW index)
    - Simple API
    - Free and open-source
    
    Architecture:
    - Collection: Like a table in SQL database
    - Documents: Text chunks with metadata
    - Embeddings: Vector representations
    - IDs: Unique identifiers
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = None,
        embedding_generator: EmbeddingGenerator = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection (like table name)
            persist_directory: Where to save the database
            embedding_generator: Generator for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.vectordb_dir
        
        # Initialize embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for RAG system"}
        )
        
        logger.info(
            f"✅ Initialized VectorStore: collection='{collection_name}', "
            f"documents={self.collection.count()}"
        )
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """
        Add document chunks to the vector store.
        
        This is the main ingestion pipeline:
        1. Extract text from chunks
        2. Generate embeddings
        3. Store in ChromaDB with metadata
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks added
            
        Example:
            >>> chunks = chunker.chunk_text(text, "doc.pdf", "doc_001")
            >>> count = vector_store.add_documents(chunks)
            >>> print(f"Added {count} chunks")
        """
        if not chunks:
            logger.warning("No chunks provided")
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract data from chunks
        texts = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        ids = [self._generate_id(chunk) for chunk in chunks]
        
        # Generate embeddings (batched for efficiency)
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.embed_batch(texts, show_progress=True)
        
        # Add to ChromaDB in batches
        logger.info("Storing in vector database...")
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            total_added += (batch_end - i)
            logger.info(f"  Stored {total_added}/{len(chunks)} chunks")
        
        logger.info(f"✅ Added {total_added} chunks to vector store")
        return total_added
    
    def search(
        self,
        query: str,
        n_results: int = None,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        This is where RAG magic happens:
        1. Convert query to embedding
        2. Find most similar document embeddings
        3. Return documents ranked by similarity
        
        Args:
            query: Search query (natural language)
            n_results: Number of results to return (default from config)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with text, metadata, and scores
            
        Example:
            >>> results = vector_store.search("What is RAG?", n_results=5)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f}")
            ...     print(f"Text: {result['text'][:100]}...")
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        n_results = n_results or settings.top_k_results
        
        logger.info(f"Searching for: '{query}' (top {n_results})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata  # Optional metadata filters
        )
        
        # Format results
        formatted_results = self._format_search_results(results)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict:
        """
        Prepare metadata for storage in ChromaDB.
        
        ChromaDB metadata constraints:
        - Values must be: str, int, float, or bool
        - No nested dictionaries
        - No None values
        
        We flatten the chunk metadata and add essential fields.
        """
        metadata = {
            "document_name": chunk.document_name,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }
        
        # Add page number if available
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        
        # Add flattened chunk metadata
        if chunk.metadata:
            for key, value in chunk.metadata.items():
                # Only add simple types
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
        
        return metadata
    
    def _generate_id(self, chunk: DocumentChunk) -> str:
        """
        Generate unique ID for a chunk.
        
        Format: {document_id}_{chunk_index}
        Example: doc_123_0, doc_123_1, doc_123_2
        
        Why this format:
        - Unique across all documents
        - Allows easy filtering by document
        - Human-readable for debugging
        """
        return f"{chunk.document_id}_{chunk.chunk_index}"
    
    def _format_search_results(self, raw_results: Dict) -> List[Dict]:
        """
        Format ChromaDB results into a clean structure.
        
        ChromaDB with cosine distance returns distances in range [0, 2].
        We convert to similarity scores [0, 1].
        """
        formatted = []
        
        # ChromaDB returns lists of lists (for batch queries)
        ids = raw_results['ids'][0] if raw_results['ids'] else []
        documents = raw_results['documents'][0] if raw_results['documents'] else []
        metadatas = raw_results['metadatas'][0] if raw_results['metadatas'] else []
        distances = raw_results['distances'][0] if raw_results['distances'] else []
        
        for i in range(len(ids)):
            # Convert cosine distance to similarity
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Similarity: 1 = identical, 0 = opposite
            distance = distances[i]
            similarity = max(0.0, 1.0 - (distance / 2.0))
            
            formatted.append({
                'id': ids[i],
                'text': documents[i],
                'metadata': metadatas[i],
                'score': similarity,  # Now correct!
                'distance': distance
            })
        
        return formatted
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks from a specific document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Number of chunks deleted
        """
        # Query for all chunks with this document_id
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if not results['ids']:
            logger.warning(f"No chunks found for document {document_id}")
            return 0
        
        # Delete the chunks
        self.collection.delete(ids=results['ids'])
        
        logger.info(f"✅ Deleted {len(results['ids'])} chunks from document {document_id}")
        return len(results['ids'])
    
    def count_documents(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()
    
    def list_documents(self) -> List[str]:
        """
        Get list of unique document IDs in the store.
        
        Returns:
            List of document IDs
        """
        # Get all metadata
        results = self.collection.get()
        
        if not results['metadatas']:
            return []
        
        # Extract unique document IDs
        document_ids = set()
        for metadata in results['metadatas']:
            if 'document_id' in metadata:
                document_ids.add(metadata['document_id'])
        
        return sorted(list(document_ids))
    
    def reset(self):
        """Delete all documents from the collection (use with caution!)"""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks for RAG system"}
        )
        logger.info("✅ Collection reset")


# Convenience function
def create_vector_store(collection_name: str = "documents") -> VectorStore:
    """
    Quick helper to create a vector store.
    
    Usage:
        store = create_vector_store()
    """
    return VectorStore(collection_name=collection_name)