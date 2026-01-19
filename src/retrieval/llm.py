"""
LLM Integration Module
Handles response generation using OpenAI's GPT models.

Key Features:
- Prompt engineering for RAG
- Citation extraction
- Streaming support
- Error handling and retries
"""

from typing import List, Dict, Optional, Generator
import logging
from openai import OpenAI
import json

from src.utils.config import settings

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Generate responses using LLM with retrieved context.
    
    This is where RAG comes together:
    - Takes retrieved documents
    - Builds a prompt
    - Calls LLM to generate answer
    - Extracts citations
    
    Why GPT-4o-mini:
    - Cost-effective: $0.15/1M input tokens
    - Fast: ~2 second response time
    - High quality: 85%+ accuracy
    - Large context: 128k tokens
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize the LLM generator.
        
        Args:
            model: Model name (default from config)
            api_key: OpenAI API key (default from config)
            temperature: Randomness (0-1, lower = more deterministic)
            max_tokens: Maximum response length
            
        Temperature explained:
        - 0.0: Deterministic (same answer every time)
        - 0.1: Slightly varied (good for factual Q&A)
        - 0.7: Creative (good for writing)
        - 1.0: Very random (not recommended for RAG)
        """
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.openai_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(
            f"Initialized LLMGenerator: model={self.model}, "
            f"temp={self.temperature}"
        )
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: str = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Generate response using LLM with retrieved context.
        
        This is the core RAG function!
        
        Args:
            query: User's question
            context_chunks: Retrieved document chunks from vector search
            system_prompt: Custom system prompt (optional)
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with:
            - answer: Generated text
            - sources: List of source citations
            - confidence: Confidence score
            - model: Model used
            
        Example:
            >>> context = vector_store.search("What is RAG?", n_results=5)
            >>> response = llm.generate_response("What is RAG?", context)
            >>> print(response['answer'])
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        if not context_chunks:
            logger.warning("No context provided, generating answer without context")
        
        # Build the prompt
        system_msg = system_prompt or self._build_system_prompt()
        user_msg = self._build_user_prompt(query, context_chunks)
        
        logger.info(f"Generating response for query: '{query[:50]}...'")
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False  # We'll add streaming later
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Calculate tokens used (for cost tracking)
            tokens_used = response.usage.total_tokens
            
            # Extract sources if requested
            sources = []
            if include_sources and context_chunks:
                sources = self._extract_sources(context_chunks)
            
            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(context_chunks)
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'model': self.model,
                'tokens_used': tokens_used,
                'finish_reason': response.choices[0].finish_reason
            }
            
            logger.info(
                f"✅ Generated response: {tokens_used} tokens, "
                f"confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_response_stream(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: str = None
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming (word-by-word).
        
        Why streaming:
        - Better UX (user sees response as it's generated)
        - Perceived as faster
        - Can stop early if needed
        
        Yields:
            Chunks of the response as they're generated
            
        Example:
            >>> for chunk in llm.generate_response_stream(query, context):
            ...     print(chunk, end='', flush=True)
        """
        system_msg = system_prompt or self._build_system_prompt()
        user_msg = self._build_user_prompt(query, context_chunks)
        
        logger.info(f"Generating streaming response for: '{query[:50]}...'")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True  # Enable streaming
            )
            
            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for RAG.
        
        This is CRITICAL for good RAG performance!
        
        Key instructions:
        1. Answer ONLY from provided context
        2. Cite sources
        3. Say "I don't know" if context insufficient
        4. Be concise but complete
        """
        return """You are a helpful AI assistant that answers questions based on provided document context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context provided below
2. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question based on the provided documents"
3. Include citations in your answer using the format [Source: document_name, Page: X]
4. Be concise but thorough - provide enough detail to fully answer the question
5. If multiple sources support your answer, cite all relevant sources
6. Do not make up information or use knowledge outside the provided context

Your goal is to provide accurate, well-cited answers that help users understand the information in their documents."""
    
    def _build_user_prompt(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> str:
        """
        Build user prompt with query and context.
        
        Format:
        CONTEXT:
        [Document 1, Page 3]
        Text from chunk 1...
        
        [Document 2, Page 5]
        Text from chunk 2...
        
        QUESTION:
        What is RAG?
        
        Why this format:
        - Clear separation of context and question
        - Source info visible to LLM for citations
        - Easy for LLM to parse
        """
        # Build context section
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            doc_name = chunk['metadata'].get('document_name', 'Unknown')
            page_num = chunk['metadata'].get('page_number', 'N/A')
            text = chunk['text']
            
            context_part = f"[Source {i}: {doc_name}, Page: {page_num}]\n{text}"
            context_parts.append(context_part)
        
        context_text = "\n\n".join(context_parts)
        
        # Build full prompt
        prompt = f"""CONTEXT:
{context_text if context_text else "No relevant context found."}

QUESTION:
{query}

Please provide a detailed answer based on the context above."""
        
        return prompt
    
    def _extract_sources(self, context_chunks: List[Dict]) -> List[Dict]:
        """
        Extract source citations from context chunks.
        
        NOW PROPERLY EXTRACTS ALL SOURCES!
        """
        sources = []
        seen = set()  # Track unique sources
        
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            doc_name = metadata.get('document_name', 'Unknown')
            page_num = metadata.get('page_number', 'N/A')
            
            # Get similarity score - CHECK BOTH PLACES
            if 'distance' in chunk:
                distance = chunk['distance']
                score = max(0.0, 1.0 - (distance / 2.0))
            elif 'distance' in metadata:
                distance = metadata['distance']
                score = max(0.0, 1.0 - (distance / 2.0))
            elif 'score' in chunk:
                score = chunk['score']
            else:
                score = 0.0
            
            # Create unique key (doc + page)
            source_key = f"{doc_name}_{page_num}"
            
            if source_key not in seen:
                sources.append({
                    'document_name': doc_name,
                    'page_number': page_num,
                    'relevance_score': score
                })
                seen.add(source_key)
        
        # Sort by relevance
        sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return sources
    
    def _calculate_confidence(self, context_chunks: List[Dict]) -> float:
        """
        Calculate confidence score based on retrieval quality.
        
        NOW USING CORRECT DISTANCE VALUES!
        """
        if not context_chunks:
            return 0.0
        
        # Extract similarity scores from chunks
        similarities = []
        for chunk in context_chunks:
            # ChromaDB returns 'distance' in chunk metadata
            if 'distance' in chunk:
                distance = chunk['distance']
                # Convert cosine distance to similarity
                # Distance: 0 = identical, 2 = opposite
                similarity = max(0.0, 1.0 - (distance / 2.0))
            elif 'score' in chunk:
                # Fallback
                similarity = chunk['score']
            else:
                # No score found
                similarity = 0.0
            
            similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Weighted average favoring top results
        if len(similarities) >= 3:
            weights = [0.5, 0.3, 0.2]
            confidence = sum(sim * weight for sim, weight in zip(similarities[:3], weights))
        elif len(similarities) == 2:
            confidence = similarities[0] * 0.6 + similarities[1] * 0.4
        else:
            confidence = similarities[0]
        
        # Boost for very strong top result
        if similarities[0] > 0.85:
            confidence = min(1.0, confidence * 1.1)
        
        # Boost for multiple high-quality results
        high_quality = sum(1 for s in similarities if s > 0.8)
        if high_quality >= 2:
            confidence = min(1.0, confidence * 1.05)
        
        return round(confidence, 3)


# Convenience function
def generate_answer(query: str, context_chunks: List[Dict]) -> Dict:
    """
    Quick helper to generate an answer.
    
    Usage:
        context = vector_store.search("What is RAG?")
        response = generate_answer("What is RAG?", context)
        print(response['answer'])
    """
    generator = LLMGenerator()
    return generator.generate_response(query, context_chunks)