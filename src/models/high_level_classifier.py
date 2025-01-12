# src/models/high_level_classifier.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Tuple
import numpy as np

class SystemsThinkingClassifier:
    def __init__(self, model_path: str, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        ).to(self.device)
        
    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Predict if text exhibits systems thinking
        Returns: (is_systems_thinking, confidence)
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction[0]].item()
            
        return bool(prediction.item()), confidence
        
    def batch_predict(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """Batch prediction for multiple texts"""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = [probabilities[i][pred].item() 
                         for i, pred in enumerate(predictions)]
            
        return list(zip(map(bool, predictions.tolist()), confidences))

# src/models/subdimension_classifier.py

class SubdimensionClassifier:
    DIMENSIONS = [
        'purpose',
        'tensions',
        'macro_issue_why',
        'macro_issue_how',
        'micro_issue_why',
        'micro_issue_how',
        'collaboration',
        'agency'
    ]
    
    def __init__(self, model_path: str, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.DIMENSIONS)
        ).to(self.device)
        
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict probabilities for each subdimension
        Returns: Dict mapping dimension names to probabilities
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            
        return {
            dim: prob.item()
            for dim, prob in zip(self.DIMENSIONS, probabilities)
        }
        
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k most likely subdimensions"""
        probs = self.predict(text)
        sorted_dims = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:k]

# src/models/embeddings_generator.py

class EmbeddingsGenerator:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.client = OpenAI()  # Assumes API key is set in environment
        self.batch_size = 32
        
    def generate(self, text: str) -> np.ndarray:
        """Generate embeddings for a single text"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
        
    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
        
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms