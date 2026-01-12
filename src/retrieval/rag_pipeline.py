"""
Complete RAG Query Pipeline
Orchestrates: Query → Retrieval → Generation → Response

This is the main interface for asking questions!
"""

from typing import Dict, List, Optional, Generator
import logging
from datetime import datetime

from src.retrieval.vectorstore import VectorStore
from src.retrieval.llm import LLMGenerator
from src.utils.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    
    This is what users interact with:
    - Ask a question
    - Get an answer with sources
    
    Under the hood:
    1. Embed query
    2. Search vector database
    3. Generate response with LLM
    4. Return formatted answer
    
    Usage:
        rag = RAGPipeline()
        response = rag.query("What is machine learning?")
        print(response['answer'])
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        llm_generator: LLMGenerator = None,
        conversation_history: List[Dict] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector database (created if not provided)
            llm_generator: LLM generator (created if not provided)
            conversation_history: Optional conversation memory
        """
        self.vector_store = vector_store or VectorStore()
        self.llm_generator = llm_generator or LLMGenerator()
        self.conversation_history = conversation_history or []
        
        logger.info("✅ Initialized RAGPipeline")
    
    def query(
        self,
        question: str,
        n_results: int = None,
        filters: Dict = None,
        include_context: bool = False
    ) -> Dict:
        """
        Ask a question and get an answer.
        
        This is the main entry point!
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve (default from config)
            filters: Optional metadata filters for retrieval
            include_context: Whether to include retrieved chunks in response
            
        Returns:
            Dictionary with:
            - answer: Generated response
            - sources: Source citations
            - confidence: Confidence score
            - model: Model used
            - tokens_used: Tokens consumed
            - retrieved_chunks: Retrieved context (if include_context=True)
            - timestamp: When query was processed
            
        Example:
            >>> rag = RAGPipeline()
            >>> response = rag.query("What is deep learning?")
            >>> print(response['answer'])
            >>> print(f"Confidence: {response['confidence']}")
            >>> for source in response['sources']:
            ...     print(f"  - {source['document_name']}, Page {source['page_number']}")
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        logger.info(f"Processing query: '{question}'")
        start_time = datetime.now()
        
        # Step 1: Retrieve relevant context
        logger.info("Step 1/2: Retrieving relevant context...")
        n_results = n_results or settings.top_k_results
        
        context_chunks = self.vector_store.search(
            query=question,
            n_results=n_results,
            filter_metadata=filters
        )
        
        if not context_chunks:
            logger.warning("No relevant context found")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'model': self.llm_generator.model,  # ✅ Fixed
                'tokens_used': 0,  # ✅ Fixed
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': int(processing_time)
            }
        
        logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
        
        # Step 2: Generate response with LLM
        logger.info("Step 2/2: Generating response...")
        
        llm_response = self.llm_generator.generate_response(
            query=question,
            context_chunks=context_chunks,
            include_sources=True
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build final response
        response = {
            'answer': llm_response['answer'],
            'sources': llm_response['sources'],
            'confidence': llm_response['confidence'],
            'model': llm_response['model'],
            'tokens_used': llm_response['tokens_used'],
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': int(processing_time)
        }
        
        # Optionally include retrieved context
        if include_context:
            response['retrieved_chunks'] = context_chunks
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'answer': llm_response['answer'],
            'timestamp': response['timestamp']
        })
        
        logger.info(
            f"✅ Query processed in {processing_time:.0f}ms, "
            f"confidence={response['confidence']:.2f}"
        )
        
        return response
    
    def query_stream(
        self,
        question: str,
        n_results: int = None,
        filters: Dict = None
    ) -> Generator[str, None, None]:
        """
        Ask a question and stream the answer word-by-word.
        
        Great for interactive UIs!
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
            filters: Optional metadata filters
            
        Yields:
            Chunks of the answer as they're generated
            
        Example:
            >>> for chunk in rag.query_stream("What is AI?"):
            ...     print(chunk, end='', flush=True)
        """
        logger.info(f"Processing streaming query: '{question}'")
        
        # Retrieve context
        n_results = n_results or settings.top_k_results
        context_chunks = self.vector_store.search(
            query=question,
            n_results=n_results,
            filter_metadata=filters
        )
        
        if not context_chunks:
            yield "I couldn't find any relevant information in the documents to answer your question."
            return
        
        # Stream response
        for chunk in self.llm_generator.generate_response_stream(
            query=question,
            context_chunks=context_chunks
        ):
            yield chunk
    
    def get_conversation_history(self, n: int = 10) -> List[Dict]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent exchanges to return
            
        Returns:
            List of recent Q&A pairs
        """
        return self.conversation_history[-n:]
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with stats about the system
        """
        total_docs = len(self.vector_store.list_documents())
        total_chunks = self.vector_store.count_documents()
        total_queries = len(self.conversation_history)
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'total_queries': total_queries,
            'avg_confidence': self._calculate_avg_confidence(),
            'documents': self.vector_store.list_documents()
        }
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence across conversation history."""
        # This would require storing confidence in history
        # For now, return placeholder
        return 0.0


# Convenience function
def ask_question(question: str) -> str:
    """
    Quick helper to ask a question and get an answer.
    
    Usage:
        answer = ask_question("What is RAG?")
        print(answer)
    """
    rag = RAGPipeline()
    response = rag.query(question)
    return response['answer']