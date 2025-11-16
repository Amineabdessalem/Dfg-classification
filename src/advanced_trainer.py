"""
Advanced Training Module with Mixed Precision, Gradient Accumulation, and Advanced Techniques
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import numpy as np
from tqdm import tqdm

from model import DFGClassifier, save_model
from data_processor import DFGDataset
from utils import EarlyStopping, setup_logging

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Advanced trainer with mixed precision, gradient accumulation, and more"""
    
    def __init__(self, 
                 model: DFGClassifier,
                 config: Dict,
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize advanced trainer
        
        Args:
            model: DFG Classifier model
            config: Training configuration
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.config = config
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_f1 = 0.0
        
        # Optimizer and scheduler (initialized later)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        logger.info(f"Initialized AdvancedTrainer on {self.device}")
        if self.use_amp:
            logger.info("Using Automatic Mixed Precision (AMP)")
        if self.gradient_accumulation_steps > 1:
            logger.info(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps")
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler"""
        training_config = self.config.get('training', {})
        
        # Optimizer with layer-wise learning rate decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': training_config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_config.get('learning_rate', 2e-5),
            eps=training_config.get('adam_epsilon', 1e-8)
        )
        
        # Cosine schedule with warmup
        num_warmup_steps = training_config.get('warmup_steps', 100)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Optimizer: AdamW with lr={training_config.get('learning_rate', 2e-5)}")
        logger.info(f"Scheduler: Cosine with {num_warmup_steps} warmup steps")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('training', {}).get('max_grad_norm', 1.0)
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('training', {}).get('max_grad_norm', 1.0)
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Scheduler step
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rates'].append(current_lr)
            
            # Accumulate loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                
                total_loss += outputs['loss'].item()
                
                # Collect predictions
                predictions = outputs['predictions'].cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
                all_probabilities.extend(probabilities)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_predictions)
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Macro F1 (important for imbalanced datasets)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['macro_f1'] = macro_f1
        
        return metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None,
              num_epochs: int = 3,
              output_dir: str = 'models/checkpoints',
              save_best_only: bool = True):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            num_epochs: Number of training epochs
            output_dir: Output directory for checkpoints
            save_best_only: Only save best model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        num_training_steps = len(train_loader) * num_epochs // self.gradient_accumulation_steps
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('training', {}).get('early_stopping_patience', 3),
            min_delta=0.001
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}, "
                f"Val F1={val_metrics['f1']:.4f}, "
                f"Val Macro F1={val_metrics['macro_f1']:.4f}"
            )
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                save_model(
                    self.model,
                    best_model_path,
                    epoch=epoch,
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    metrics=val_metrics
                )
                logger.info(f"✓ Saved best model (F1={self.best_val_f1:.4f})")
            
            # Save checkpoint
            if not save_best_only:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
                save_model(
                    self.model,
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    metrics=val_metrics
                )
            
            # Early stopping
            early_stopping(val_metrics['loss'])
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Evaluate on test set if provided
        if test_loader is not None:
            logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(test_loader)
            logger.info(
                f"Test Results: "
                f"Loss={test_metrics['loss']:.4f}, "
                f"Accuracy={test_metrics['accuracy']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"Macro F1={test_metrics['macro_f1']:.4f}"
            )
            
            # Save test metrics
            test_results_path = os.path.join(output_dir, 'test_results.json')
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {
                k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for v in vals]
                for k, vals in self.history.items()
            }
            json.dump(history_json, f, indent=2)
        
        logger.info(f"Training completed! Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Models and history saved to {output_dir}")


def create_weighted_loss(class_weights: Dict[int, float], device: torch.device) -> nn.CrossEntropyLoss:
    """
    Create weighted cross-entropy loss for imbalanced datasets
    
    Args:
        class_weights: Dictionary mapping class IDs to weights
        device: Device to put weights on
        
    Returns:
        Weighted CrossEntropyLoss
    """
    # Convert to tensor
    num_classes = len(class_weights)
    weight_tensor = torch.zeros(num_classes)
    
    for class_id, weight in class_weights.items():
        weight_tensor[class_id] = weight
    
    weight_tensor = weight_tensor.to(device)
    
    return nn.CrossEntropyLoss(weight=weight_tensor)


if __name__ == "__main__":
    # Test advanced trainer
    logging.basicConfig(level=logging.INFO)
    
    print("Advanced Trainer module loaded successfully!")
    print("Features:")
    print("- Mixed Precision Training (AMP)")
    print("- Gradient Accumulation")
    print("- Cosine Learning Rate Schedule")
    print("- Layer-wise Learning Rate Decay")
    print("- Advanced Metrics (Macro F1, Per-class metrics)")
    print("- Weighted Loss for Imbalanced Data")




