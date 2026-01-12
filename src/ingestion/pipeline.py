"""
Complete Document Ingestion Pipeline
Orchestrates: Parsing → Chunking → Embedding → Storage

This ties everything together into one easy-to-use pipeline.
"""

from typing import List, Optional
import logging
from pathlib import Path

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vectorstore import VectorStore
from src.utils.config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    
    Usage:
        pipeline = IngestionPipeline()
        pipeline.ingest_document("paper.pdf")
        
    What it does:
        1. Parse document (PDF/DOCX/TXT)
        2. Chunk text intelligently
        3. Generate embeddings
        4. Store in vector database
        
    All in one line!
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        parser: DocumentParser = None,
        chunker: TextChunker = None
    ):
        """Initialize the pipeline with optional custom components."""
        self.vector_store = vector_store or VectorStore()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or TextChunker()
        
        logger.info("✅ Initialized IngestionPipeline")
    
    def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None
    ) -> dict:
        """
        Ingest a single document into the system.
        
        Args:
            file_path: Path to document file
            document_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            Dictionary with ingestion statistics
            
        Example:
            >>> pipeline = IngestionPipeline()
            >>> result = pipeline.ingest_document("research_paper.pdf")
            >>> print(f"Processed {result['num_chunks']} chunks")
        """
        logger.info(f"Starting ingestion: {file_path}")
        
        # Step 1: Parse document
        logger.info("Step 1/4: Parsing document...")
        text, metadata = self.parser.parse(file_path)
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(file_path)
        
        # Step 2: Chunk text
        logger.info("Step 2/4: Chunking text...")
        chunks = self.chunker.chunk_text(
            text=text,
            document_name=Path(file_path).name,
            document_id=document_id,
            metadata=metadata
        )
        
        if not chunks:
            logger.warning("No chunks generated from document")
            return {
                'success': False,
                'document_id': document_id,
                'num_chunks': 0,
                'error': 'No chunks generated'
            }
        
        # Step 3 & 4: Generate embeddings and store
        logger.info("Step 3/4: Generating embeddings...")
        logger.info("Step 4/4: Storing in vector database...")
        num_added = self.vector_store.add_documents(chunks)
        
        # Summary
        result = {
            'success': True,
            'document_id': document_id,
            'document_name': Path(file_path).name,
            'num_chunks': len(chunks),
            'num_stored': num_added,
            'metadata': metadata
        }
        
        logger.info(f"✅ Ingestion complete: {result}")
        return result
    
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = False
    ) -> List[dict]:
        """
        Ingest all supported documents in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of results (one per document)
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Find all supported files
        pattern = '**/*' if recursive else '*'
        files = []
        for ext in self.parser.SUPPORTED_EXTENSIONS:
            files.extend(path.glob(f"{pattern}{ext}"))
        
        logger.info(f"Found {len(files)} documents in {directory_path}")
        
        # Ingest each file
        results = []
        for file_path in files:
            try:
                result = self.ingest_document(str(file_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {str(e)}")
                results.append({
                    'success': False,
                    'document_name': file_path.name,
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"✅ Ingested {successful}/{len(results)} documents")
        
        return results
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate a unique document ID from file path."""
        import hashlib
        
        # Use file path hash as ID
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        filename = Path(file_path).stem
        
        return f"doc_{filename}_{path_hash}"


# Convenience function
def ingest_document(file_path: str) -> dict:
    """
    Quick helper to ingest a document.
    
    Usage:
        result = ingest_document("paper.pdf")
    """
    pipeline = IngestionPipeline()
    return pipeline.ingest_document(file_path)