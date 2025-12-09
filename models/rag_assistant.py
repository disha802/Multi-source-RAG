# ============================================================================
# models/rag_assistant.py
# ============================================================================
"""
RAG Assistant - handles queries and answer generation
"""
import time
import hashlib
from typing import Dict
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

class RAGAssistant:
    """Main RAG assistant with caching"""
    
    def __init__(self, vector_store, config):
        self.vector_store = vector_store
        self.config = config
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.cache = {}
    
    def query(self, question: str) -> Dict:
        """Process a query and return answer with metadata"""
        from utils.helpers import validate_query, calculate_similarity
        
        start_time = time.time()
        
        # Validate
        is_valid, error = validate_query(question)
        if not is_valid:
            return {'error': error}
        
        # Check cache
        cache_key = hashlib.md5(question.lower().strip().encode()).hexdigest()
        if cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['cached'] = True
            result['latency'] = round(time.time() - start_time, 3)
            return result
        
        # Check if vector store is ready
        if not self.vector_store.is_loaded:
            return {'error': 'Vector database not initialized. Upload documents first.'}
        
        # Search for relevant docs
        results = self.vector_store.search(question)
        
        if not results:
            return {'error': 'No relevant documents found'}
        
        # Build context
        context_parts = []
        sources = []
        similarities = []
        
        for i, (doc, distance) in enumerate(results):
            similarity = calculate_similarity(distance)
            similarities.append(similarity)
            
            context_parts.append(
                f"[SOURCE {i+1}] (Relevance: {similarity:.1%})\n{doc.page_content}\n"
            )
            
            sources.append({
                'id': i + 1,
                'title': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'type': doc.metadata.get('type', 'Unknown'),
                'relevance': round(similarity, 3)
            })
        
        context = "\n".join(context_parts)
        
        # Generate answer
        try:
            answer = self._generate_answer(context, question)
        except Exception as e:
            return {'error': f'Failed to generate answer: {str(e)}'}
        
        # Prepare result
        result = {
            'answer': answer,
            'sources': sources,
            'avg_similarity': round(sum(similarities) / len(similarities), 3),
            'num_sources': len(sources),
            'latency': round(time.time() - start_time, 3),
            'cached': False
        }
        
        # Cache it
        self.cache[cache_key] = result.copy()
        if len(self.cache) > 100:
            self.cache.pop(next(iter(self.cache)))
        
        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using LLM with retry logic"""
        prompt = f"""Answer the question using ONLY the provided sources.

Rules:
- Cite sources: "According to Source 1..."
- If info is missing, say "Not available in documents"
- Do not add external knowledge

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""
        
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    def clear_cache(self):
        """Clear query cache"""
        self.cache.clear()
