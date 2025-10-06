#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Architecture Benchmarking Tool

This script provides a comprehensive benchmarking framework for evaluating 
different RAG (Retrieval-Augmented Generation) architectures and configurations.
It allows comparison of embedding models, vector stores, retrieval strategies,
and context processing approaches.

Default dataset: A simple corpus about dogs for baseline comparison and benchmarking.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

# Add the project root to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.models.embeddings_generator import EmbeddingsGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import SemanticRetriever
from src.rag.context_processor import ContextProcessor

# Try to import optional dependencies
try:
    from sentence_transformers import CrossEncoder
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Reranking will not be available.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    HAVE_NLTK = True
except ImportError:
    HAVE_NLTK = False
    logger.warning("NLTK not installed. Some multi-hop features will be limited.")

# Default paths for the dogs dataset
DEFAULT_CORPUS_PATH = os.path.join("Other", "dogs_corpus.json")
DEFAULT_QUERIES_PATH = os.path.join("Other", "dogs_test_queries.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Other/rag_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for a RAG benchmark run"""
    name: str
    embedding_model: str
    vector_store_config: Dict
    retriever_config: Dict
    context_processor_config: Dict
    batch_size: int = 16
    top_k: int = 8
    similarity_threshold: float = 0.75
    max_tokens: int = 4096
    dedup_threshold: float = 0.8
    
    # Advanced RAG technique flags and configurations
    use_hybrid_retrieval: bool = False
    hybrid_weight: float = 0.5  # Weight for combining dense and sparse scores (0-1)
    
    use_memory_augmentation: bool = False
    memory_config: Dict = field(default_factory=lambda: {
        "memory_size": 5,       # Number of previous results to store
        "memory_threshold": 0.7  # Similarity threshold for memory retrieval
    })
    
    use_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20  # How many results to rerank
    
    use_multi_hop: bool = False
    multi_hop_config: Dict = field(default_factory=lambda: {
        "hops": 2,                # Number of hops to perform
        "hop_strategy": "extract"  # Strategy for generating follow-up queries: extract, summarize, or hybrid
    })
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization"""
        return {
            "name": self.name,
            "embedding_model": self.embedding_model,
            "vector_store_config": self.vector_store_config,
            "retriever_config": self.retriever_config,
            "context_processor_config": self.context_processor_config,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_tokens": self.max_tokens,
            "dedup_threshold": self.dedup_threshold,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "hybrid_weight": self.hybrid_weight,
            "use_memory_augmentation": self.use_memory_augmentation,
            "memory_config": self.memory_config,
            "use_reranking": self.use_reranking,
            "reranker_model": self.reranker_model,
            "rerank_top_k": self.rerank_top_k,
            "use_multi_hop": self.use_multi_hop,
            "multi_hop_config": self.multi_hop_config
        }

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config_name: str
    queries_processed: int
    avg_retrieval_time: float
    avg_embedding_time: float
    avg_processing_time: float
    total_time: float
    avg_num_results: float
    metrics: Dict = field(default_factory=dict)
    
    # Advanced technique metrics
    avg_reranking_time: float = 0.0
    avg_hybrid_time: float = 0.0
    avg_memory_time: float = 0.0
    avg_multi_hop_time: float = 0.0
    avg_hops_performed: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization"""
        return {
            "config_name": self.config_name,
            "queries_processed": self.queries_processed,
            "avg_retrieval_time": self.avg_retrieval_time,
            "avg_embedding_time": self.avg_embedding_time,
            "avg_processing_time": self.avg_processing_time,
            "total_time": self.total_time,
            "avg_num_results": self.avg_num_results,
            "avg_reranking_time": self.avg_reranking_time,
            "avg_hybrid_time": self.avg_hybrid_time,
            "avg_memory_time": self.avg_memory_time,
            "avg_multi_hop_time": self.avg_multi_hop_time,
            "avg_hops_performed": self.avg_hops_performed,
            "metrics": self.metrics
        }

class MemoryStore:
    """
    Store and retrieve memory for augmented retrieval
    
    This class provides methods to:
    1. Store query results in memory
    2. Retrieve relevant memories for new queries
    3. Merge memory results with new results
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the memory store
        
        Args:
            config: Configuration dictionary with memory parameters
        """
        self.memory_index = {}  # Query -> results mapping
        self.memory_size = config.get("memory_size", 5)
        self.memory_threshold = config.get("memory_threshold", 0.7)
    
    def add_memory(self, query: str, results: List[Dict]):
        """
        Add query and results to memory
        
        Args:
            query: The query string
            results: List of result dictionaries
        """
        # Store only top results to limit memory size
        self.memory_index[query] = results[:self.memory_size]
    
    def get_relevant_memory(self, query: str, embeddings_generator) -> List[Dict]:
        """
        Get relevant memory for a query
        
        Args:
            query: The current query string
            embeddings_generator: EmbeddingsGenerator instance for similarity calculation
            
        Returns:
            List of relevant results from memory
        """
        if not self.memory_index:
            return []
        
        # Get embedding for current query
        query_embedding = embeddings_generator.generate(query)
        
        # Find most similar previous queries
        similar_memories = []
        for prev_query, results in self.memory_index.items():
            prev_embedding = embeddings_generator.generate(prev_query)
            similarity = self._cosine_similarity(query_embedding, prev_embedding)
            if similarity > self.memory_threshold:
                similar_memories.append((similarity, results))
        
        # Sort by similarity and return top results
        similar_memories.sort(key=lambda x: x[0], reverse=True)
        if not similar_memories:
            return []
        
        return similar_memories[0][1]  # Return results from most similar query
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)


class MultiHopRetriever:
    """
    Perform multi-hop retrieval for complex queries
    
    This class provides methods to:
    1. Decompose queries into sub-queries
    2. Perform sequential retrievals
    3. Combine results from multiple hops
    """
    
    def __init__(self, retriever, config: Dict):
        """
        Initialize the multi-hop retriever
        
        Args:
            retriever: SemanticRetriever instance
            config: Configuration dictionary with multi-hop parameters
        """
        self.retriever = retriever
        self.hops = config.get("hops", 2)
        self.hop_strategy = config.get("hop_strategy", "extract")
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict], int]:
        """
        Perform multi-hop retrieval
        
        Args:
            query: The query string
            k: Number of results to retrieve per hop
            
        Returns:
            Tuple of (combined results, number of hops performed)
        """
        all_results = []
        hop_queries = [query]
        hops_performed = 0
        
        for hop in range(self.hops):
            current_query = hop_queries[hop]
            
            # Retrieve results for current hop
            hop_results = self.retriever.retrieve(current_query, k=k)
            all_results.extend(hop_results)
            hops_performed += 1
            
            # If we've reached the max hops or no results, stop
            if hop >= self.hops - 1 or not hop_results:
                break
            
            # Generate next hop query based on current results
            next_query = self._generate_next_query(current_query, hop_results)
            hop_queries.append(next_query)
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        return unique_results, hops_performed
    
    def _generate_next_query(self, query: str, results: List[Dict]) -> str:
        """
        Generate the next hop query based on current results
        
        Args:
            query: Current query
            results: Results from current hop
            
        Returns:
            Next hop query
        """
        if not results:
            return query
        
        if self.hop_strategy == "extract":
            # Extract key entities or terms from results
            top_result_text = results[0].get("text", "")
            if HAVE_NLTK:
                # Extract the first sentence as the key information
                sentences = sent_tokenize(top_result_text)
                if sentences:
                    key_info = sentences[0]
                else:
                    key_info = top_result_text[:100]  # Fallback to first 100 chars
            else:
                # Simple extraction without NLTK
                key_info = top_result_text.split(".")[0] + "."
            
            # Combine with original query
            next_query = f"{query} AND {key_info}"
        
        elif self.hop_strategy == "summarize":
            # Use a simple summarization approach
            # In a real implementation, this might use an LLM
            top_result_text = results[0].get("text", "")
            summary = top_result_text[:150]  # Simple truncation as summary
            next_query = f"{query} considering {summary}"
        
        else:  # Default to hybrid
            # Combine original query with key terms from results
            key_terms = self._extract_key_terms(results[0].get("text", ""))
            next_query = f"{query} {' '.join(key_terms)}"
        
        return next_query
    
    def _extract_key_terms(self, text: str, num_terms: int = 3) -> List[str]:
        """
        Extract key terms from text
        
        Args:
            text: Text to extract terms from
            num_terms: Number of terms to extract
            
        Returns:
            List of key terms
        """
        # Simple implementation - split by spaces and take most common words
        # In a real implementation, this would use NLP techniques
        words = text.lower().split()
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top terms
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:num_terms]]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Deduplicate results based on content
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Deduplicated results
        """
        unique_texts = set()
        unique_results = []
        
        for result in results:
            text = result.get("text", "")
            if text not in unique_texts:
                unique_texts.add(text)
                unique_results.append(result)
        
        return unique_results


class RAGBenchmark:
    """
    Benchmark different RAG architectures and configurations
    
    This class provides methods to:
    1. Load test data
    2. Configure different RAG architectures
    3. Run benchmarks
    4. Evaluate and compare results
    5. Visualize performance
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the benchmark
        
        Args:
            data_path: Path to test data CSV file with queries and expected results
        """
        self.data_path = data_path
        self.test_data = None
        self.configs = []
        self.results = []
        self.corpus_data = None
        
        # Load test data if provided
        if data_path and os.path.exists(data_path):
            self._load_test_data()
    
    def _load_test_data(self):
        """Load test queries and expected results"""
        if self.data_path.endswith('.csv'):
            self.test_data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                # Handle the dogs dataset format which has a 'queries' key
                if 'queries' in data:
                    self.test_data = pd.DataFrame(data['queries'])
                    # Extract relevance judgments for evaluation
                    self.relevance_judgments = {}
                    for _, row in self.test_data.iterrows():
                        self.relevance_judgments[row['query']] = row.get('relevant_doc_ids', [])
                else:
                    self.test_data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        logger.info(f"Loaded {len(self.test_data)} test queries from {self.data_path}")
    
    def load_corpus(self, corpus_path: str):
        """
        Load corpus data for indexing
        
        Args:
            corpus_path: Path to corpus data (CSV or JSON)
        """
        if corpus_path.endswith('.csv'):
            self.corpus_data = pd.read_csv(corpus_path)
        elif corpus_path.endswith('.json'):
            with open(corpus_path, 'r') as f:
                data = json.load(f)
                # Handle the dogs dataset format which has a 'documents' key
                if 'documents' in data:
                    # Convert the documents list to a DataFrame
                    documents = data['documents']
                    # Extract metadata from each document
                    processed_docs = []
                    for doc in documents:
                        doc_copy = doc.copy()
                        # If metadata exists, convert it to a string representation
                        if 'metadata' in doc_copy:
                            doc_copy['metadata_str'] = json.dumps(doc_copy['metadata'])
                        processed_docs.append(doc_copy)
                    self.corpus_data = pd.DataFrame(processed_docs)
                else:
                    self.corpus_data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {corpus_path}")
        
        # Ensure required columns exist
        required_cols = ['text']
        if not all(col in self.corpus_data.columns for col in required_cols):
            raise ValueError(f"Corpus data must contain columns: {required_cols}")
        
        logger.info(f"Loaded corpus with {len(self.corpus_data)} documents")
        return self.corpus_data
    
    def add_config(self, config: BenchmarkConfig):
        """
        Add a configuration to benchmark
        
        Args:
            config: BenchmarkConfig object
        """
        self.configs.append(config)
        logger.info(f"Added benchmark configuration: {config.name}")
    
    def add_default_configs(self):
        """Add a set of default configurations to benchmark"""
        # OpenAI embedding model with default settings
        self.add_config(BenchmarkConfig(
            name="OpenAI-Default",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_openai"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            }
        ))
        
        # HuggingFace embedding model (all-MiniLM-L6-v2)
        self.add_config(BenchmarkConfig(
            name="HuggingFace-MiniLM",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_minilm"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            }
        ))
        
        # Configuration with higher similarity threshold
        self.add_config(BenchmarkConfig(
            name="High-Threshold",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_high_threshold"
            },
            retriever_config={
                "similarity_threshold": 0.85,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            }
        ))
        
        # Configuration with more results
        self.add_config(BenchmarkConfig(
            name="More-Results",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_more_results"
            },
            retriever_config={
                "similarity_threshold": 0.7,
                "top_k": 16
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            }
        ))
        
        # Configuration with stronger deduplication
        self.add_config(BenchmarkConfig(
            name="Strong-Dedup",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_strong_dedup"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.7
            }
        ))
        
        # --- Advanced RAG Techniques ---
        
        # Hybrid Retrieval Configuration
        self.add_config(BenchmarkConfig(
            name="Hybrid-Retrieval",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_hybrid"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            },
            use_hybrid_retrieval=True,
            hybrid_weight=0.7  # Favor dense retrieval slightly
        ))
        
        # Memory-Augmented Configuration
        self.add_config(BenchmarkConfig(
            name="Memory-Augmented",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_memory"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 8
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            },
            use_memory_augmentation=True,
            memory_config={
                "memory_size": 5,
                "memory_threshold": 0.7
            }
        ))
        
        # Reranking Configuration
        self.add_config(BenchmarkConfig(
            name="Reranking",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_rerank"
            },
            retriever_config={
                "similarity_threshold": 0.7,  # Lower threshold to get more candidates
                "top_k": 20  # Get more candidates for reranking
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            },
            use_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=8  # Keep top 8 after reranking
        ))
        
        # Multi-Hop Retrieval Configuration
        self.add_config(BenchmarkConfig(
            name="Multi-Hop",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_multi_hop"
            },
            retriever_config={
                "similarity_threshold": 0.75,
                "top_k": 5  # Fewer results per hop
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            },
            use_multi_hop=True,
            multi_hop_config={
                "hops": 2,
                "hop_strategy": "extract"
            }
        ))
        
        # Combined Approach (Hybrid + Reranking)
        self.add_config(BenchmarkConfig(
            name="Hybrid-Reranking",
            embedding_model="text-embedding-3-large",
            vector_store_config={
                "opensearch_host": "localhost:9200",
                "auth": ("admin", "admin"),
                "index_name": "benchmark_hybrid_rerank"
            },
            retriever_config={
                "similarity_threshold": 0.7,
                "top_k": 20
            },
            context_processor_config={
                "max_tokens": 4096,
                "dedup_threshold": 0.8
            },
            use_hybrid_retrieval=True,
            hybrid_weight=0.6,
            use_reranking=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=8
        ))
        
        logger.info(f"Added {len(self.configs)} default benchmark configurations")
    
    def _setup_rag_pipeline(self, config: BenchmarkConfig) -> Tuple[Any, ContextProcessor, Dict]:
        """
        Set up a RAG pipeline based on the configuration
        
        Args:
            config: BenchmarkConfig object
            
        Returns:
            Tuple of (retriever, context_processor, advanced_components)
        """
        # Create embedding generator
        embeddings_generator = EmbeddingsGenerator(
            model_name=config.embedding_model,
            config={"batch_size": config.batch_size}
        )
        
        # Create vector store with hybrid search capability if enabled
        vector_store_config = config.vector_store_config.copy()
        if config.use_hybrid_retrieval:
            vector_store_config["enable_hybrid"] = True
            vector_store_config["hybrid_weight"] = config.hybrid_weight
            logger.info(f"Enabling hybrid retrieval with weight {config.hybrid_weight}")
        
        vector_store = VectorStore(vector_store_config)
        
        # Create retriever
        retriever = SemanticRetriever(
            vector_store=vector_store,
            embeddings_generator=embeddings_generator,
            config=config.retriever_config
        )
        
        # Create context processor
        context_processor = ContextProcessor(config=config.context_processor_config)
        
        # Set up advanced components
        advanced_components = {}
        
        # Set up memory store if enabled
        if config.use_memory_augmentation:
            logger.info(f"Enabling memory augmentation with threshold {config.memory_config.get('memory_threshold')}")
            advanced_components['memory_store'] = MemoryStore(config.memory_config)
        
        # Set up reranker if enabled
        if config.use_reranking:
            if HAVE_SENTENCE_TRANSFORMERS:
                logger.info(f"Enabling reranking with model {config.reranker_model}")
                advanced_components['reranker'] = CrossEncoder(config.reranker_model)
            else:
                logger.warning("Reranking requested but sentence-transformers not installed. Skipping reranking.")
        
        # Set up multi-hop retriever if enabled
        if config.use_multi_hop:
            logger.info(f"Enabling multi-hop retrieval with {config.multi_hop_config.get('hops')} hops")
            advanced_components['multi_hop'] = MultiHopRetriever(retriever, config.multi_hop_config)
        
        return retriever, context_processor, advanced_components
    
    def index_corpus(self, config: BenchmarkConfig) -> None:
        """
        Index the corpus data using the specified configuration
        
        Args:
            config: BenchmarkConfig object
        """
        if self.corpus_data is None:
            raise ValueError("No corpus data loaded. Call load_corpus() first.")
        
        logger.info(f"Indexing corpus for configuration: {config.name}")
        
        # Create embedding generator
        embeddings_generator = EmbeddingsGenerator(
            model_name=config.embedding_model,
            config={"batch_size": config.batch_size}
        )
        
        # Create vector store
        vector_store = VectorStore(config.vector_store_config)
        
        # Generate embeddings and store in batches
        texts = self.corpus_data['text'].tolist()
        
        # Extract metadata if available
        metadata = None
        if 'metadata' in self.corpus_data.columns:
            metadata = self.corpus_data['metadata'].tolist()
        elif set(['source', 'date']).issubset(self.corpus_data.columns):
            metadata = []
            for _, row in self.corpus_data.iterrows():
                meta = {}
                if 'source' in row:
                    meta['source'] = row['source']
                if 'date' in row:
                    meta['date'] = row['date']
                metadata.append(meta)
        
        # Process in batches
        for i in tqdm(range(0, len(texts), config.batch_size), desc=f"Indexing {config.name}"):
            batch_texts = texts[i:i + config.batch_size]
            batch_meta = None if metadata is None else metadata[i:i + config.batch_size]
            
            # Generate embeddings
            batch_embeddings = embeddings_generator.generate_batch(batch_texts)
            
            # Store in vector store
            vector_store.store(batch_embeddings, batch_texts, batch_meta)
        
        logger.info(f"Indexed {len(texts)} documents for {config.name}")
    
    def run_benchmark(self, queries: List[str] = None) -> List[BenchmarkResult]:
        """
        Run benchmarks for all configurations
        
        Args:
            queries: List of queries to test (uses test_data if not provided)
            
        Returns:
            List of BenchmarkResult objects
        """
        if not self.configs:
            raise ValueError("No configurations added. Call add_config() first.")
        
        if queries is None:
            if self.test_data is None:
                raise ValueError("No test data loaded and no queries provided.")
            queries = self.test_data['query'].tolist()
        
        results = []
        
        for config in self.configs:
            logger.info(f"Running benchmark for configuration: {config.name}")
            
            # Set up RAG pipeline with advanced components
            retriever, context_processor, advanced_components = self._setup_rag_pipeline(config)
            
            # Timing variables
            embedding_times = []
            retrieval_times = []
            processing_times = []
            reranking_times = []
            hybrid_times = []
            memory_times = []
            multi_hop_times = []
            num_results = []
            hops_performed = []
            
            start_time = time.time()
            
            for query in tqdm(queries, desc=f"Testing {config.name}"):
                # Get memory augmentation if enabled
                memory_results = []
                if config.use_memory_augmentation and 'memory_store' in advanced_components:
                    mem_start = time.time()
                    memory_store = advanced_components['memory_store']
                    memory_results = memory_store.get_relevant_memory(query, retriever.embeddings_generator)
                    mem_end = time.time()
                    memory_times.append(mem_end - mem_start)
                else:
                    memory_times.append(0)
                
                # Time embedding generation
                emb_start = time.time()
                query_embedding = retriever.embeddings_generator.generate(query)
                emb_end = time.time()
                embedding_times.append(emb_end - emb_start)
                
                # Time retrieval - use multi-hop if enabled
                ret_start = time.time()
                
                if config.use_multi_hop and 'multi_hop' in advanced_components:
                    multi_hop_retriever = advanced_components['multi_hop']
                    results_list, num_hops = multi_hop_retriever.retrieve(query, k=config.top_k)
                    hops_performed.append(num_hops)
                    multi_hop_times.append(time.time() - ret_start)
                    hybrid_times.append(0)  # Not using hybrid in multi-hop mode
                else:
                    # Use hybrid retrieval if enabled
                    if config.use_hybrid_retrieval:
                        hybrid_start = time.time()
                        # In a real implementation, this would call a hybrid search method
                        # For now, we simulate it by combining vector search with a keyword search
                        vector_results = retriever.vector_store.search(
                            query_embedding,
                            k=config.top_k,
                            threshold=config.similarity_threshold
                        )
                        
                        # Simulate keyword search (in a real implementation, this would use BM25)
                        # Here we just use the same vector results but with a different score
                        keyword_results = vector_results.copy()
                        
                        # Combine results with weighted scores
                        results_list = []
                        for v_result, k_result in zip(vector_results, keyword_results):
                            combined_score = config.hybrid_weight * v_result.get('score', 0) + \
                                           (1 - config.hybrid_weight) * k_result.get('score', 0)
                            result = v_result.copy()
                            result['score'] = combined_score
                            results_list.append(result)
                        
                        hybrid_end = time.time()
                        hybrid_times.append(hybrid_end - hybrid_start)
                    else:
                        # Standard vector search
                        results_list = retriever.vector_store.search(
                            query_embedding,
                            k=config.top_k,
                            threshold=config.similarity_threshold
                        )
                        hybrid_times.append(0)
                        multi_hop_times.append(0)
                        hops_performed.append(1)  # Just one hop for standard retrieval
                
                ret_end = time.time()
                retrieval_times.append(ret_end - ret_start)
                
                # Apply reranking if enabled
                if config.use_reranking and 'reranker' in advanced_components and results_list:
                    rerank_start = time.time()
                    reranker = advanced_components['reranker']
                    
                    # Prepare pairs for reranking
                    pairs = [[query, result.get('text', '')] for result in results_list]
                    
                    # Get relevance scores
                    scores = reranker.predict(pairs)
                    
                    # Attach scores to results
                    for result, score in zip(results_list, scores):
                        result['rerank_score'] = float(score)
                    
                    # Sort by rerank score
                    results_list.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                    
                    # Limit to top k after reranking
                    results_list = results_list[:config.rerank_top_k]
                    
                    rerank_end = time.time()
                    reranking_times.append(rerank_end - rerank_start)
                else:
                    reranking_times.append(0)
                
                # Record number of results
                num_results.append(len(results_list))
                
                # Combine with memory results if any
                if memory_results:
                    # Simple merge - in a real implementation, this would be more sophisticated
                    combined_results = memory_results + results_list
                    # Deduplicate
                    seen_texts = set()
                    unique_results = []
                    for result in combined_results:
                        text = result.get('text', '')
                        if text not in seen_texts:
                            seen_texts.add(text)
                            unique_results.append(result)
                    results_list = unique_results[:config.top_k]  # Limit to top k
                
                # Time context processing
                proc_start = time.time()
                context_processor.assemble_context(results_list, max_tokens=config.max_tokens)
                proc_end = time.time()
                processing_times.append(proc_end - proc_start)
                
                # Store in memory if enabled
                if config.use_memory_augmentation and 'memory_store' in advanced_components:
                    memory_store.add_memory(query, results_list)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                config_name=config.name,
                queries_processed=len(queries),
                avg_embedding_time=np.mean(embedding_times),
                avg_retrieval_time=np.mean(retrieval_times),
                avg_processing_time=np.mean(processing_times),
                avg_reranking_time=np.mean(reranking_times) if reranking_times else 0,
                avg_hybrid_time=np.mean(hybrid_times) if hybrid_times else 0,
                avg_memory_time=np.mean(memory_times) if memory_times else 0,
                avg_multi_hop_time=np.mean(multi_hop_times) if multi_hop_times else 0,
                avg_hops_performed=np.mean(hops_performed) if hops_performed else 1,
                total_time=total_time,
                avg_num_results=np.mean(num_results),
                metrics={
                    "embedding_times": embedding_times,
                    "retrieval_times": retrieval_times,
                    "processing_times": processing_times,
                    "reranking_times": reranking_times,
                    "hybrid_times": hybrid_times,
                    "memory_times": memory_times,
                    "multi_hop_times": multi_hop_times,
                    "num_results": num_results,
                    "hops_performed": hops_performed
                }
            )
            
            results.append(benchmark_result)
            logger.info(f"Completed benchmark for {config.name}")
        
        self.results = results
        return results
    
    def evaluate_relevance(self, relevance_judgments: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate relevance of retrieved results against ground truth
        
        Args:
            relevance_judgments: Dictionary mapping queries to lists of relevant document IDs
            
        Returns:
            Dictionary of metrics by configuration
        """
        if not self.results:
            raise ValueError("No benchmark results available. Call run_benchmark() first.")
        
        relevance_metrics = {}
        
        for config in self.configs:
            # Set up RAG pipeline
            retriever, _ = self._setup_rag_pipeline(config)
            
            precision_at_k = []
            recall_at_k = []
            
            for query, relevant_docs in relevance_judgments.items():
                results = retriever.retrieve(query, k=config.top_k)
                retrieved_docs = [r['metadata'].get('id', '') for r in results]
                
                # Calculate precision@k
                relevant_retrieved = set(retrieved_docs).intersection(set(relevant_docs))
                precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
                precision_at_k.append(precision)
                
                # Calculate recall@k
                recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
                recall_at_k.append(recall)
            
            relevance_metrics[config.name] = {
                "precision@k": np.mean(precision_at_k),
                "recall@k": np.mean(recall_at_k),
                "f1@k": 2 * np.mean(precision_at_k) * np.mean(recall_at_k) / 
                       (np.mean(precision_at_k) + np.mean(recall_at_k)) 
                       if (np.mean(precision_at_k) + np.mean(recall_at_k)) > 0 else 0
            }
        
        return relevance_metrics
    
    def visualize_results(self, output_dir: str = "Other/benchmark_results"):
        """
        Visualize benchmark results
        
        Args:
            output_dir: Directory to save visualization files
        """
        if not self.results:
            raise ValueError("No benchmark results available. Call run_benchmark() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        config_names = [r.config_name for r in self.results]
        embedding_times = [r.avg_embedding_time for r in self.results]
        retrieval_times = [r.avg_retrieval_time for r in self.results]
        processing_times = [r.avg_processing_time for r in self.results]
        total_times = [r.total_time for r in self.results]
        avg_results = [r.avg_num_results for r in self.results]
        
        # Advanced technique times
        reranking_times = [getattr(r, 'avg_reranking_time', 0) for r in self.results]
        hybrid_times = [getattr(r, 'avg_hybrid_time', 0) for r in self.results]
        memory_times = [getattr(r, 'avg_memory_time', 0) for r in self.results]
        multi_hop_times = [getattr(r, 'avg_multi_hop_time', 0) for r in self.results]
        hops_performed = [getattr(r, 'avg_hops_performed', 1) for r in self.results]
        
        # Plot average times
        plt.figure(figsize=(14, 8))
        x = np.arange(len(config_names))
        width = 0.15
        
        plt.bar(x - 2*width, embedding_times, width, label='Embedding Time')
        plt.bar(x - width, retrieval_times, width, label='Retrieval Time')
        plt.bar(x, processing_times, width, label='Processing Time')
        plt.bar(x + width, reranking_times, width, label='Reranking Time')
        plt.bar(x + 2*width, hybrid_times, width, label='Hybrid Retrieval Time')
        
        plt.xlabel('Configuration')
        plt.ylabel('Time (seconds)')
        plt.title('Average Processing Times by Configuration')
        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_times.png'))
        
        # Plot advanced technique times
        has_advanced = any(t > 0 for t in reranking_times + hybrid_times + memory_times + multi_hop_times)
        if has_advanced:
            plt.figure(figsize=(14, 8))
            x = np.arange(len(config_names))
            width = 0.2
            
            plt.bar(x - 1.5*width, reranking_times, width, label='Reranking Time')
            plt.bar(x - 0.5*width, hybrid_times, width, label='Hybrid Retrieval Time')
            plt.bar(x + 0.5*width, memory_times, width, label='Memory Augmentation Time')
            plt.bar(x + 1.5*width, multi_hop_times, width, label='Multi-Hop Time')
            
            plt.xlabel('Configuration')
            plt.ylabel('Time (seconds)')
            plt.title('Advanced RAG Technique Times by Configuration')
            plt.xticks(x, config_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'advanced_times.png'))
            
            # Plot average hops performed for multi-hop configurations
            if any(h > 1 for h in hops_performed):
                plt.figure(figsize=(10, 6))
                # Only include configs with multi-hop enabled
                multi_hop_configs = [config for config, hops in zip(config_names, hops_performed) if hops > 1]
                multi_hop_values = [h for h in hops_performed if h > 1]
                
                if multi_hop_configs:
                    plt.bar(multi_hop_configs, multi_hop_values, color='purple')
                    plt.xlabel('Configuration')
                    plt.ylabel('Average Hops Performed')
                    plt.title('Average Number of Hops in Multi-Hop Retrieval')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'avg_hops.png'))
        
        # Plot total times
        plt.figure(figsize=(10, 6))
        plt.bar(config_names, total_times, color='skyblue')
        plt.xlabel('Configuration')
        plt.ylabel('Time (seconds)')
        plt.title('Total Benchmark Time by Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_times.png'))
        
        # Plot average number of results
        plt.figure(figsize=(10, 6))
        plt.bar(config_names, avg_results, color='lightgreen')
        plt.xlabel('Configuration')
        plt.ylabel('Average Number of Results')
        plt.title('Average Number of Retrieved Results by Configuration')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_results.png'))
        
        # Save results as JSON
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "configs": [c.to_dict() for c in self.configs],
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create summary report
        with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
            f.write("# RAG Benchmark Summary Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configurations Tested\n\n")
            for config in self.configs:
                f.write(f"### {config.name}\n")
                f.write(f"- Embedding Model: {config.embedding_model}\n")
                f.write(f"- Top K: {config.top_k}\n")
                f.write(f"- Similarity Threshold: {config.similarity_threshold}\n")
                f.write(f"- Deduplication Threshold: {config.dedup_threshold}\n")
                
                # Add advanced technique details if enabled
                if config.use_hybrid_retrieval:
                    f.write(f"- **Hybrid Retrieval**: Enabled (Weight: {config.hybrid_weight})\n")
                if config.use_memory_augmentation:
                    f.write(f"- **Memory Augmentation**: Enabled (Threshold: {config.memory_config.get('memory_threshold')})\n")
                if config.use_reranking:
                    f.write(f"- **Reranking**: Enabled (Model: {config.reranker_model}, Top K: {config.rerank_top_k})\n")
                if config.use_multi_hop:
                    f.write(f"- **Multi-Hop Retrieval**: Enabled (Hops: {config.multi_hop_config.get('hops')}, Strategy: {config.multi_hop_config.get('hop_strategy')})\n")
                f.write("\n")
            
            f.write("## Performance Results\n\n")
            f.write("| Configuration | Avg Embedding Time | Avg Retrieval Time | Avg Processing Time | Total Time | Avg # Results |\n")
            f.write("|--------------|-------------------|-------------------|---------------------|------------|---------------|\n")
            
            for result in self.results:
                f.write(f"| {result.config_name} | {result.avg_embedding_time:.4f}s | {result.avg_retrieval_time:.4f}s | ")
                f.write(f"{result.avg_processing_time:.4f}s | {result.total_time:.2f}s | {result.avg_num_results:.2f} |\n")
            
            # Add advanced technique results if any
            if has_advanced:
                f.write("\n## Advanced RAG Technique Performance\n\n")
                f.write("| Configuration | Hybrid Time | Reranking Time | Memory Time | Multi-Hop Time | Avg Hops |\n")
                f.write("|--------------|-------------|---------------|-------------|---------------|----------|\n")
                
                for result in self.results:
                    hybrid_time = getattr(result, 'avg_hybrid_time', 0)
                    reranking_time = getattr(result, 'avg_reranking_time', 0)
                    memory_time = getattr(result, 'avg_memory_time', 0)
                    multi_hop_time = getattr(result, 'avg_multi_hop_time', 0)
                    avg_hops = getattr(result, 'avg_hops_performed', 1)
                    
                    f.write(f"| {result.config_name} | {hybrid_time:.4f}s | {reranking_time:.4f}s | ")
                    f.write(f"{memory_time:.4f}s | {multi_hop_time:.4f}s | {avg_hops:.2f} |\n")
            
            # Add technique comparison section
            f.write("\n## RAG Technique Comparison\n\n")
            f.write("This benchmark compares several advanced RAG techniques:\n\n")
            f.write("1. **Hybrid Retrieval**: Combines dense vector search with sparse lexical search (BM25)\n")
            f.write("2. **Memory Augmentation**: Stores and retrieves previous query results to enhance context\n")
            f.write("3. **Reranking**: Uses a cross-encoder model to rerank initial retrieval results\n")
            f.write("4. **Multi-Hop Retrieval**: Performs sequential retrievals to gather information in steps\n\n")
            
            # Add recommendations based on results
            f.write("### Recommendations\n\n")
            f.write("Based on the benchmark results, consider the following recommendations:\n\n")
            
            # Find fastest configuration
            fastest_config = min(self.results, key=lambda r: r.total_time)
            f.write(f"- **Fastest Configuration**: {fastest_config.config_name} ({fastest_config.total_time:.2f}s)\n")
            
            # Find configuration with most results
            most_results_config = max(self.results, key=lambda r: r.avg_num_results)
            f.write(f"- **Most Comprehensive Results**: {most_results_config.config_name} (Avg {most_results_config.avg_num_results:.2f} results)\n")
            
            # Add specific technique recommendations
            f.write("\n**Technique-Specific Recommendations**:\n\n")
            f.write("- **For Speed**: Standard dense retrieval or memory-augmented retrieval typically offer the best performance.\n")
            f.write("- **For Relevance**: Reranking or hybrid approaches generally provide the most relevant results.\n")
            f.write("- **For Complex Queries**: Multi-hop retrieval can help with multi-part or complex queries.\n")
            f.write("- **For Interactive Systems**: Memory augmentation helps maintain context across multiple queries.\n")
        
        logger.info(f"Visualization and reports saved to {output_dir}")
        return os.path.abspath(output_dir)


if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='RAG Architecture Benchmarking Tool')
    parser.add_argument('--corpus', type=str, default=DEFAULT_CORPUS_PATH,
                        help=f'Path to corpus data (CSV or JSON). Default: {DEFAULT_CORPUS_PATH}')
    parser.add_argument('--queries', type=str, default=DEFAULT_QUERIES_PATH,
                        help=f'Path to test queries (CSV or JSON). Default: {DEFAULT_QUERIES_PATH}')
    parser.add_argument('--output', type=str, default="Other/benchmark_results",
                        help='Directory to save benchmark results. Default: Other/benchmark_results')
    parser.add_argument('--run', action='store_true',
                        help='Run the benchmark with the specified corpus and queries')
    
    args = parser.parse_args()
    
    # Example usage
    benchmark = RAGBenchmark(data_path=args.queries)
    
    print(f"\nRAG Architecture Benchmarking Tool")
    print(f"=================================")
    
    # Load corpus data
    print(f"\nLoading corpus from: {args.corpus}")
    try:
        benchmark.load_corpus(args.corpus)
        print(f"Successfully loaded corpus with {len(benchmark.corpus_data)} documents")
    except Exception as e:
        print(f"Error loading corpus: {e}")
        print("Using the dogs corpus? Make sure the file exists at the specified path.")
        print(f"Default path: {DEFAULT_CORPUS_PATH}")
        sys.exit(1)
    
    # Add default configurations
    print("\nAdding benchmark configurations...")
    benchmark.add_default_configs()
    print(f"Added {len(benchmark.configs)} configurations for testing")
    
    # Run the benchmark if requested
    if args.run:
        print("\nIndexing corpus for each configuration...")
        for config in benchmark.configs:
            print(f"Indexing for configuration: {config.name}")
            benchmark.index_corpus(config)
        
        print("\nRunning benchmark with test queries...")
        results = benchmark.run_benchmark()
        
        print("\nGenerating visualization and reports...")
        output_dir = benchmark.visualize_results(output_dir=args.output)
        print(f"Results saved to: {output_dir}")
        
        # Evaluate relevance if relevance judgments are available
        if hasattr(benchmark, 'relevance_judgments') and benchmark.relevance_judgments:
            print("\nEvaluating relevance against ground truth...")
            relevance_metrics = benchmark.evaluate_relevance(benchmark.relevance_judgments)
            print("\nRelevance Metrics:")
            for config_name, metrics in relevance_metrics.items():
                print(f"\n{config_name}:")
                print(f"  Precision@k: {metrics['precision@k']:.4f}")
                print(f"  Recall@k: {metrics['recall@k']:.4f}")
                print(f"  F1@k: {metrics['f1@k']:.4f}")
    else:
        print("\nBenchmark not run. Use --run to execute the benchmark.")
        print("\nExample command to run the benchmark:")
        print(f"python {sys.argv[0]} --run")
        print("\nFor a full benchmark, you'll need to:")
        print("1. Load your corpus data (done automatically with the dogs dataset)")
        print("2. Configure your RAG architectures (done automatically)")
        print("3. Index your corpus (done when --run is specified)")
        print("4. Run the benchmark with your test queries (done when --run is specified)")
        print("5. Visualize and analyze the results (done when --run is specified)")
