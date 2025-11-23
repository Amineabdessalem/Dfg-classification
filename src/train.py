"""
Training pipeline for DFG Subject Area Classifier
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm

from model import DFGClassifier, save_model
from data_processor import (
    DFGDatasetProcessor, DFGDataset, load_config, load_dfg_mapping
)
from utils import setup_logging, EarlyStopping

# Set up logging
logger = logging.getLogger(__name__)


class Trainer:
    """Main training class for DFG classifier"""
    
    def __init__(self, config: Dict, dfg_mapping: Dict):
        self.config = config
        self.dfg_mapping = dfg_mapping
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', {}).get('use_cuda', True) else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize data processor
        self.data_processor = DFGDatasetProcessor(config, dfg_mapping)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def _initialize_model(self) -> DFGClassifier:
        """Initialize the model"""
        model_config = self.config.get('model', {})
        
        model = DFGClassifier(
            model_name=model_config.get('name', 'allenai/scibert_scivocab_uncased'),
            num_classes=model_config.get('num_classes', 30),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            freeze_bert=model_config.get('freeze_bert', False),
            hierarchical=model_config.get('hierarchical', False),
            dfg_mapping=self.dfg_mapping,
            allowed_labels=model_config.get('allowed_labels')
        )
        
        model.to(self.device)
        logger.info(f"Model initialized: {model.get_model_info()}")
        
        return model
    
    def _initialize_optimizer_and_scheduler(self, num_training_steps: int):
        """Initialize optimizer and scheduler"""
        training_config = self.config.get('training', {})
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        num_warmup_steps = training_config.get('warmup_steps', 100)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Optimizer and scheduler initialized")
    
    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data
        
        Args:
            data_path: Path to processed dataset or raw data directory
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Check if processed dataset exists
        processed_path = os.path.join(data_path, "processed_dataset.json")
        
        if os.path.exists(processed_path):
            logger.info("Loading processed dataset...")
            with open(processed_path, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Convert back to tensors
            dataset = []
            for item in dataset_data:
                dataset.append({
                    'input_ids': torch.tensor(item['input_ids']),
                    'attention_mask': torch.tensor(item['attention_mask']),
                    'labels': torch.tensor(item['labels']),
                    'filename': item['filename'],
                    'title': item['title'],
                    'abstract': item['abstract'],
                    'label': item['label']
                })
        else:
            logger.info("Processing raw data...")
            # This would process raw PDFs - for now, create sample data
            dataset = self._create_sample_dataset()
        
        # Split dataset
        train_data, val_data, test_data = self.data_processor.split_dataset(dataset)
        
        logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)
        val_loader = self._create_data_loader(val_data, shuffle=False)
        test_loader = self._create_data_loader(test_data, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _create_sample_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """Create sample dataset for testing"""
        import random
        
        dataset = []
        label_codes = list(self.data_processor.label_to_id.keys())
        
        for i in range(num_samples):
            # Create random input
            input_ids = torch.randint(0, 1000, (512,))
            attention_mask = torch.ones(512)
            
            # Random label
            label_code = random.choice(label_codes)
            label_id = self.data_processor.label_to_id[label_code]
            
            dataset.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label_id),
                'filename': f"sample_{i}.pdf",
                'title': f"Sample Paper {i}",
                'abstract': f"This is a sample abstract for paper {i}",
                'label': label_code
            })
        
        return dataset
    
    def _create_data_loader(self, dataset: List[Dict], shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader"""
        torch_dataset = DFGDataset(dataset)
        
        device_config = self.config.get('device', {})
        batch_size = self.config.get('training', {}).get('batch_size', 8)
        
        return DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=device_config.get('num_workers', 4),
            pin_memory=device_config.get('pin_memory', True)
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            training_config = self.config.get('training', {})
            max_grad_norm = training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                predictions = outputs['predictions'].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, data_path: str, output_dir: str):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data(data_path)
        
        # Initialize optimizer and scheduler
        num_training_steps = len(train_loader) * self.config.get('training', {}).get('num_epochs', 3)
        self._initialize_optimizer_and_scheduler(num_training_steps)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('training', {}).get('early_stopping_patience', 3),
            min_delta=0.001
        )
        
        # Training loop
        num_epochs = self.config.get('training', {}).get('num_epochs', 3)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy, val_f1 = self.validate(val_loader)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_f1'].append(val_f1)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.get('training', {}).get('save_best_model', True):
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    save_model(
                        self.model,
                        best_model_path,
                        epoch=epoch,
                        optimizer_state_dict=self.optimizer.state_dict(),
                        scheduler_state_dict=self.scheduler.state_dict(),
                        metrics={
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'val_f1': val_f1
                        }
                    )
                    logger.info(f"Best model saved at epoch {epoch}")
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model.pt")
        save_model(
            self.model,
            final_model_path,
            epoch=epoch,
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            metrics=self.training_history
        )
        
        # Evaluate on test set
        test_loss, test_accuracy, test_f1 = self.validate(test_loader)
        logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train DFG Subject Area Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints", help="Output directory for models")
    parser.add_argument("--dfg_mapping", type=str, default="data/dfg_mapping.json", help="Path to DFG mapping file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load DFG mapping
    dfg_mapping = load_dfg_mapping(args.dfg_mapping)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(config, dfg_mapping)
    
    # Train model
    trainer.train(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()

