from typing import Dict, List, Optional
from .vector_store import VectorStore
from ..models.embeddings_generator import EmbeddingsGenerator

class SemanticRetriever:
    def __init__(self, vector_store: VectorStore, 
                 embeddings_generator: EmbeddingsGenerator,
                 config: Dict = None):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)
        self.top_k = self.config.get('top_k', 8)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant contexts for a query"""
        k = k or self.top_k
        query_embedding = self.embeddings_generator.generate(query)
        
        results = self.vector_store.search(
            query_embedding,
            k=k,
            threshold=self.similarity_threshold
        )
        
        return results

    def batch_retrieve(self, queries: List[str], k: Optional[int] = None) -> List[List[Dict]]:
        """Retrieve contexts for multiple queries"""
        k = k or self.top_k
        query_embeddings = self.embeddings_generator.generate_batch(queries)
        
        all_results = []
        for embedding in query_embeddings:
            results = self.vector_store.search(
                embedding,
                k=k,
                threshold=self.similarity_threshold
            )
            all_results.append(results)
            
        return all_results