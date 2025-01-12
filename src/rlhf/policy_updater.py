# src/rlhf/policy_updater.py

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

class PolicyUpdater:
    def __init__(self, model_path: str, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5)
        )
        
    def update_policy(self, feedback_batch: List[Feedback]):
        """Update model based on expert feedback"""
        # Group feedback by dimension
        dimension_feedback = defaultdict(list)
        for feedback in feedback_batch:
            if feedback.dimension:
                dimension_feedback[feedback.dimension].append(feedback)
                
        # Process each dimension separately
        for dimension, feedbacks in dimension_feedback.items():
            self._update_dimension_policy(dimension, feedbacks)
            
    def _update_dimension_policy(self, dimension: str, feedbacks: List[Feedback]):
        """Update policy for a specific dimension"""
        # Prepare training data
        texts = []
        ratings = []
        
        for feedback in feedbacks:
            texts.append(feedback.text_id)  # Assuming text_id contains actual text
            # Convert 1-5 rating to training signal
            rating = (feedback.rating - 1) / 4.0  # Normalize to [0,1]
            ratings.append(rating)
            
        # Convert to tensors
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        ratings_tensor = torch.tensor(ratings).float().to(self.device)
        
        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()
        
        # MSE loss for regression
        loss = nn.MSELoss()(logits, ratings_tensor)
        
        loss.backward()
        self.optimizer.step()
        
    def save_policy(self, path: str):
        """Save updated model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)