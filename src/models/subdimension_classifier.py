# src/models/subdimension_classifier.py

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import numpy as np

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
        
        # Load label encodings
        self.label2id = {dim: i for i, dim in enumerate(self.DIMENSIONS)}
        self.id2label = {i: dim for dim, i in self.label2id.items()}
        
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
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            
        return {
            dim: prob.item()
            for dim, prob in zip(self.DIMENSIONS, probabilities)
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict subdimensions for multiple texts"""
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        results = []
        for probs in probabilities:
            results.append({
                dim: prob.item()
                for dim, prob in zip(self.DIMENSIONS, probs)
            })
        return results
        
    def get_top_dimensions(self, predictions: Dict[str, float], 
                          k: int = 3, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Get top k predicted dimensions above threshold"""
        filtered_preds = {
            dim: prob for dim, prob in predictions.items()
            if prob >= threshold
        }
        sorted_dims = sorted(filtered_preds.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:k]