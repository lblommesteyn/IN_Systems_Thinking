# src/rag/vector_store.py

from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import Dict, List, Optional
import numpy as np
import json

class VectorStore:
    def __init__(self, config: Dict):
        self.client = OpenSearch(
            hosts=[config['opensearch_host']],
            http_auth=config['auth'],
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        self.index_name = config['index_name']
        self.setup_index()
        
    def setup_index(self):
        """Create index with appropriate mappings if it doesn't exist"""
        if not self.client.indices.exists(self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "text": {"type": "text"},
                        "metadata": {"type": "object"}
                    }
                },
                "settings": {
                    "index": {
                        "number_of_shards": 2,
                        "number_of_replicas": 1,
                        "refresh_interval": "1s"
                    }
                }
            }
            self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
    def store(self, embeddings: np.ndarray, texts: List[str], 
              metadata: Optional[List[Dict]] = None):
        """Store embeddings and associated text/metadata"""
        if metadata is None:
            metadata = [{} for _ in texts]
            
        actions = []
        for emb, text, meta in zip(embeddings, texts, metadata):
            action = {
                "_index": self.index_name,
                "_source": {
                    "embedding": emb.tolist(),
                    "text": text,
                    "metadata": meta
                }
            }
            actions.append(action)
            
        # Bulk index
        from opensearchpy.helpers import bulk
        bulk(self.client, actions)
        
    def search(self, query_embedding: np.ndarray, k: int = 8, 
               threshold: float = 0.75) -> List[Dict]:
        """
        Search for similar vectors
        Returns: List of {text, score, metadata} dicts
        """
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
        
        response = self.client.search(
            index=self.index_name,
            body={
                "size": k,
                "query": script_query,
                "_source": ["text", "metadata"],
                "min_score": threshold
            }
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'text': hit['_source']['text'],
                'score': hit['_score'],
                'metadata': hit['_source']['metadata']
            })
            
        return results

# src/rag/retriever.py

from typing import Dict, List, Optional
import numpy as np
from .vector_store import VectorStore
from ..models.embeddings_generator import EmbeddingsGenerator

class SemanticRetriever:
    def __init__(self, vector_store: VectorStore, 
                 embeddings_generator: EmbeddingsGenerator,
                 config: Dict = None):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.config = config or {}
        
    def retrieve(self, query: str, k: int = 8) -> List[Dict]:
        """
        Retrieve relevant contexts for a query
        Returns: List of {text, score, metadata} dicts
        """
        # Generate query embedding
        query_embedding = self.embeddings_generator.generate(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            k=k,
            threshold=self.config.get('similarity_threshold', 0.75)
        )
        
        return results
        
    def batch_retrieve(self, queries: List[str], k: int = 8) -> List[List[Dict]]:
        """Retrieve contexts for multiple queries"""
        # Generate embeddings for all queries
        query_embeddings = self.embeddings_generator.generate_batch(queries)
        
        # Search for each query
        all_results = []
        for embedding in query_embeddings:
            results = self.vector_store.search(
                embedding,
                k=k,
                threshold=self.config.get('similarity_threshold', 0.75)
            )
            all_results.append(results)
            
        return all_results

# src/rag/context_processor.py

class ContextProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def assemble_context(self, retrieved_docs: List[Dict], 
                        max_tokens: int = 4096) -> str:
        """
        Assemble retrieved documents into a single context
        with dynamic window sizing
        """
        # Sort by score
        docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
        
        context_parts = []
        total_tokens = 0
        
        for doc in docs:
            # Rough token count estimation
            doc_tokens = len(doc['text'].split())
            if total_tokens + doc_tokens > max_tokens:
                break
                
            # Add document with metadata
            context_part = f"\nSource: {doc['metadata'].get('source', 'Unknown')}"
            if 'date' in doc['metadata']:
                context_part += f" ({doc['metadata']['date']})"
            context_part += f"\n{doc['text']}\n"
            
            context_parts.append(context_part)
            total_tokens += doc_tokens
            
        return "\n".join(context_parts)
        
    def deduplicate_contexts(self, contexts: List[str], 
                           similarity_threshold: float = 0.8) -> List[str]:
        """Remove near-duplicate contexts"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not contexts:
            return []
            
        # Convert texts to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contexts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Track indices to keep
        keep_indices = set()
        for i in range(len(contexts)):
            # If this index isn't similar to any previously kept index, keep it
            if not any(similarities[i][j] > similarity_threshold 
                      for j in keep_indices):
                keep_indices.add(i)
                
        return [contexts[i] for i in sorted(keep_indices)]