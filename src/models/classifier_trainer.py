# src/models/classifier_trainer.py

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from datetime import datetime

class SystemsThinkingDataset(Dataset):
    """Dataset for systems thinking classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(label)
        
        return encoding

class MultiLabelDataset(Dataset):
    """Dataset for multi-label subdimension classification"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer):
        self.texts = texts
        self.labels = labels  # Multi-hot encoded labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.float)
        
        return encoding

def compute_metrics(pred):
    """Compute metrics for model evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_multilabel_metrics(pred):
    """Compute metrics for multi-label classification"""
    labels = pred.label_ids
    # Apply sigmoid and set threshold at 0.5
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).int().numpy()
    
    # Calculate metrics for each class
    results = {}
    for i in range(labels.shape[1]):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[:, i], preds[:, i], average='binary')
        results[f'class_{i}_f1'] = f1
        results[f'class_{i}_precision'] = precision
        results[f'class_{i}_recall'] = recall
    
    # Calculate micro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.flatten(), preds.flatten(), average='binary')
    results['micro_f1'] = f1
    results['micro_precision'] = precision
    results['micro_recall'] = recall
    
    return results

class ClassifierTrainer:
    """
    Handles model training for systems thinking classification models
    as described in Step 6 of the research approach.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the classifier trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Set parameters from config
        self.base_model = config.get('base_model', 'distilbert-base-uncased')
        self.output_dir = config.get('output_dir', './models')
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.num_epochs = config.get('num_epochs', 10)
        self.validation_split = config.get('validation_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train_high_level_classifier(self, 
                                   texts: List[str], 
                                   labels: List[int]) -> str:
        """
        Train binary classifier for systems thinking detection
        
        Args:
            texts: List of paragraph texts
            labels: Binary labels (1 for systems thinking, 0 for not)
            
        Returns:
            Path to saved model
        """
        self.logger.info("Training high-level systems thinking classifier")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2
        )
        
        # Create dataset
        dataset = SystemsThinkingDataset(texts, labels, tokenizer)
        
        # Split dataset
        train_size = int((1 - self.validation_split - self.test_split) * len(dataset))
        val_size = int(self.validation_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Set up training arguments
        model_name = f"systems_thinking_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, model_name),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_result = trainer.evaluate(test_dataset)
        self.logger.info(f"Evaluation results: {eval_result}")
        
        # Save metrics
        with open(os.path.join(self.output_dir, f"{model_name}_metrics.json"), 'w') as f:
            json.dump(eval_result, f)
        
        # Save model and tokenizer
        model_path = os.path.join(self.output_dir, model_name)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        return model_path
    
    def train_subdimension_classifier(self, 
                                     texts: List[str], 
                                     labels: List[List[int]]) -> str:
        """
        Train multi-label classifier for subdimension classification
        
        Args:
            texts: List of paragraph texts
            labels: Multi-hot encoded labels (e.g., [0, 1, 0, 0, 1, 0, 0, 0])
            
        Returns:
            Path to saved model
        """
        self.logger.info("Training subdimension classifier")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels[0]),
            problem_type="multi_label_classification"
        )
        
        # Create dataset
        dataset = MultiLabelDataset(texts, labels, tokenizer)
        
        # Split dataset
        train_size = int((1 - self.validation_split - self.test_split) * len(dataset))
        val_size = int(self.validation_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Set up training arguments
        model_name = f"subdimension_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, model_name),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_multilabel_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_result = trainer.evaluate(test_dataset)
        self.logger.info(f"Evaluation results: {eval_result}")
        
        # Save metrics
        with open(os.path.join(self.output_dir, f"{model_name}_metrics.json"), 'w') as f:
            json.dump(eval_result, f)
        
        # Save model and tokenizer
        model_path = os.path.join(self.output_dir, model_name)
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        return model_path
        
    def load_training_data(self, data_path: str) -> Tuple[List[str], List[int], Optional[List[List[int]]]]:
        """
        Load training data from CSV file
        
        Args:
            data_path: Path to CSV file with hand-coded data
            
        Returns:
            Tuple of (texts, binary_labels, subdimension_labels)
        """
        df = pd.read_csv(data_path)
        
        # Extract text and binary labels
        texts = df['text'].tolist()
        binary_labels = df['is_systems_thinking'].astype(int).tolist()
        
        # Extract subdimension labels if available
        subdim_columns = [col for col in df.columns if col.startswith('subdim_')]
        
        if subdim_columns:
            subdim_labels = df[subdim_columns].values.tolist()
            return texts, binary_labels, subdim_labels
        else:
            return texts, binary_labels, None
