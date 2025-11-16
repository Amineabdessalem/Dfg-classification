"""
Utility functions for DFG Subject Area Classifier
"""

import os
import json
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get appropriate device for computation"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def calculate_metrics(y_true: List[int], y_pred: List[int], class_names: List[str] = None) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix
    )
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }
    
    # Detailed classification report
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['detailed_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 plot
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate plot (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: List[str], save_path: Optional[str] = None):
    """
    Plot class distribution
    
    Args:
        labels: List of class labels
        save_path: Optional path to save plot
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    
    plt.figure(figsize=(15, 8))
    
    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels_sorted, counts = zip(*sorted_labels)
    
    # Create bar plot
    bars = plt.bar(range(len(labels_sorted)), counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def create_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str
):
    """
    Create comprehensive evaluation report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save report
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Create detailed report
    report = {
        'summary': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        },
        'detailed_report': metrics.get('detailed_report', {}),
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    # Save report
    save_json(report, output_path)
    
    # Create visualizations
    base_path = os.path.splitext(output_path)[0]
    
    # Confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, f"{base_path}_confusion_matrix.png")
    
    print(f"Evaluation report saved to {output_path}")


def format_prediction_output(
    prediction: Dict,
    class_names: List[str] = None,
    include_probabilities: bool = True,
    top_k: int = 5
) -> Dict:
    """
    Format prediction output for better readability
    
    Args:
        prediction: Raw prediction dictionary
        class_names: Optional class names
        include_probabilities: Whether to include probabilities
        top_k: Number of top predictions to include
        
    Returns:
        Formatted prediction dictionary
    """
    formatted = {
        'predicted_class': prediction.get('prediction_code', 'Unknown'),
        'predicted_name': prediction.get('prediction_name', 'Unknown'),
        'confidence': float(prediction.get('probabilities', [0])[prediction.get('prediction_id', 0)])
    }
    
    if include_probabilities and 'probabilities' in prediction:
        probs = prediction['probabilities']
        if class_names and len(class_names) == len(probs):
            # Get top-k predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                top_predictions.append({
                    'class_code': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                    'class_name': prediction.get('prediction_name', 'Unknown'),
                    'probability': float(probs[idx])
                })
            
            formatted['top_predictions'] = top_predictions
    
    return formatted


def validate_config(config: Dict) -> bool:
    """
    Validate configuration file
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['name', 'num_classes']
    for key in required_model_keys:
        if key not in model_config:
            print(f"Missing required model parameter: {key}")
            return False
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['learning_rate', 'batch_size', 'num_epochs']
    for key in required_training_keys:
        if key not in training_config:
            print(f"Missing required training parameter: {key}")
            return False
    
    return True


def get_model_size(model: torch.nn.Module) -> Dict:
    """
    Get model size information
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb
    }


def print_model_summary(model: torch.nn.Module):
    """Print model summary"""
    size_info = get_model_size(model)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {size_info['total_parameters']:,}")
    print(f"Trainable parameters: {size_info['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {size_info['non_trainable_parameters']:,}")
    print(f"Model size: {size_info['model_size_mb']:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test logging setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logging setup successful")
    
    # Test device detection
    device = get_device()
    print(f"Detected device: {device}")
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    print("Utility functions test completed!")

