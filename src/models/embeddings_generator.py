# src/models/embeddings_generator.py

import numpy as np
from typing import List, Dict
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingsGenerator:
    def __init__(self, model_name: str = "text-embedding-3-large", config: Dict = None):
        self.config = config or {}
        if model_name.startswith('text-embedding'):
            self.client = OpenAI()
            self.model_name = model_name
            self.use_openai = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.use_openai = False
        self.batch_size = self.config.get('batch_size', 32)
        
    def generate(self, text: str) -> np.ndarray:
        """Generate embeddings for a single text"""
        if self.use_openai:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return np.array(response.data[0].embedding)
        else:
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if self.use_openai:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings = [data.embedding for data in response.data]
            else:
                inputs = self.tokenizer(batch, return_tensors='pt',
                                      truncation=True, max_length=512, 
                                      padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)
