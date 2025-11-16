"""
Simple Training Script for DFG Classifier
CPU-friendly version with command line arguments
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processor import DFGDataset, load_config, load_dfg_mapping
from model import DFGClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_processed_dataset(data_path: str, split: str = 'train') -> list:
    """Load processed dataset"""
    file_path = os.path.join(data_path, f'{split}.json')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert back to tensors
    dataset = []
    for item in data:
        dataset.append({
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels']),
            'filename': item['filename'],
            'title': item['title'],
            'abstract': item['abstract'],
            'label': item['label']
        })
    
    return dataset


def create_data_loader(dataset: list, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Create PyTorch DataLoader"""
    torch_dataset = DFGDataset(dataset)
    
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 for CPU
        pin_memory=False  # False for CPU
    )


def train_simple(data_path: str, output_dir: str, batch_size: int = 4, num_epochs: int = 3, learning_rate: float = 0.00002):
    """Simple training function"""
    
    logger.info("🚀 Starting Simple Training...")
    logger.info(f"📊 Settings: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
    
    # Load configuration and DFG mapping
    config = load_config('config.yaml')
    dfg_mapping = load_dfg_mapping('data/dfg_mapping.json')
    
    # Load datasets
    logger.info("📊 Loading datasets...")
    train_dataset = load_processed_dataset(data_path, 'train')
    val_dataset = load_processed_dataset(data_path, 'val')
    
    logger.info(f"📈 Train: {len(train_dataset)} samples")
    logger.info(f"📈 Val: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size, shuffle=False)
    
    # Initialize model
    logger.info("🧠 Initializing model...")
    model = DFGClassifier(
        model_name='allenai/scibert_scivocab_uncased',
        num_classes=30,
        dropout_rate=0.3,
        freeze_bert=False,
        hierarchical=False,
        dfg_mapping=dfg_mapping
    )
    
    device = torch.device('cpu')  # Force CPU
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model has {total_params:,} total parameters")
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("🏋️ Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"📅 Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = outputs['predictions']
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        logger.info(f"📉 Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                predictions = outputs['predictions']
                
                val_loss += loss.item()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        logger.info(f"📊 Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"💾 Saved best model with accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    
    # Save training results
    results = {
        'final_train_loss': avg_train_loss,
        'final_train_accuracy': train_accuracy,
        'final_val_loss': avg_val_loss,
        'final_val_accuracy': val_accuracy,
        'final_val_f1': val_f1,
        'best_val_accuracy': best_val_acc,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✅ Training completed!")
    logger.info(f"🎯 Final Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"🎯 Final F1 Score: {val_f1:.4f}")
    logger.info(f"💾 Models saved to: {output_dir}")
    logger.info(f"📊 Results saved to: {results_path}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Training for DFG Classifier")
    
    parser.add_argument('--data_path', type=str, default='dfg-classifier/data/small', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='dfg-classifier/models/small', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00002, help='Learning rate')
    
    args = parser.parse_args()
    
    # Run training
    results = train_simple(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Final Accuracy: {results['final_val_accuracy']:.4f}")
    print(f"✅ Best Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"✅ F1 Score: {results['final_val_f1']:.4f}")
    print(f"📊 Total Parameters: {results['total_parameters']:,}")
    print(f"💾 Model saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


