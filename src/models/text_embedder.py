# src/models/text_embedder.py

import numpy as np
from typing import List, Dict, Union
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel
import logging

class TextEmbedder:
    """
    Handles text tokenization and embedding generation as described in Step 3
    of the Systems Thinking research approach.
    
    This class supports both OpenAI embeddings and local HuggingFace models.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-large", config: Dict = None):
        """
        Initialize the text embedder with specified model
        
        Args:
            model_name: Name of embedding model (OpenAI or HuggingFace model)
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Set up embedder based on model type
        if model_name.startswith('text-embedding'):
            self.client = OpenAI()
            self.use_openai = True
            self.embedding_dim = 3072 if '3' in model_name else 1536  # Dimensionality varies by model
            self.logger.info(f"Using OpenAI embedding model: {model_name}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.use_openai = False
            self.embedding_dim = self.model.config.hidden_size
            self.logger.info(f"Using local embedding model: {model_name}")
            
        self.batch_size = self.config.get('batch_size', 32)
    
    def tokenize(self, text: str) -> Dict:
        """
        Tokenize text using the appropriate tokenizer
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tokenizer outputs (depends on model type)
        """
        if self.use_openai:
            # OpenAI handles tokenization internally
            return {"input_text": text}
        else:
            return self.tokenizer(
                text, 
                truncation=True, 
                max_length=512, 
                padding=True, 
                return_tensors='pt'
            )
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text input
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(text, str):
            return self.embed_single(text)
        else:
            return self.embed_batch(text)
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embedding
        """
        if self.use_openai:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                self.logger.error(f"Error generating OpenAI embedding: {str(e)}")
                raise
        else:
            inputs = self.tokenize(text)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling - take average of all tokens
            return outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if self.use_openai:
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    embeddings = [np.array(data.embedding) for data in response.data]
                except Exception as e:
                    self.logger.error(f"Error generating OpenAI embeddings batch: {str(e)}")
                    raise
            else:
                inputs = self.tokenize(batch)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for improved similarity comparison
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            L2 normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
