# src/rag/context_processor.py

from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContextProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_tokens = self.config.get('max_tokens', 4096)
        self.dedup_threshold = self.config.get('dedup_threshold', 0.8)
        
    def assemble_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Assemble retrieved documents into a single context
        with dynamic window sizing
        """
        # Sort by relevance score
        docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
        
        # Deduplicate similar contexts
        unique_docs = self.deduplicate_contexts([doc['text'] for doc in docs])
        
        context_parts = []
        total_tokens = 0
        
        for doc_text in unique_docs:
            # Estimate tokens (rough approximation)
            tokens = len(doc_text.split())
            if total_tokens + tokens > self.max_tokens:
                break
                
            # Add document with metadata
            context_parts.append(doc_text)
            total_tokens += tokens
            
        return "\n\n".join(context_parts)
        
    def deduplicate_contexts(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate contexts using TF-IDF similarity"""
        if not texts:
            return []
            
        # Convert texts to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Keep track of indices to keep
        keep_indices = []
        for i in range(len(texts)):
            # If this text isn't too similar to any kept text, keep it
            if not any(similarities[i][j] > self.dedup_threshold 
                      for j in keep_indices):
                keep_indices.append(i)
                
        return [texts[i] for i in keep_indices]
        
    def rerank_contexts(self, query: str, contexts: List[str], 
                       top_k: int = 3) -> List[str]:
        """Rerank contexts based on relevance to query"""
        # Create TF-IDF vectors for query and contexts
        vectorizer = TfidfVectorizer()
        context_vectors = vectorizer.fit_transform(contexts)
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, context_vectors)[0]
        
        # Get top k contexts
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [contexts[i] for i in top_indices]
