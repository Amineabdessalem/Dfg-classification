"""
Advanced Training Script for DFG Classifier
Uses mixed precision, gradient accumulation, and advanced techniques
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model import DFGClassifier
from data_processor import DFGDataset, load_config, load_dfg_mapping
from advanced_trainer import AdvancedTrainer
from utils import setup_logging, set_seed

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


def create_data_loader(dataset: list, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """Create PyTorch DataLoader"""
    torch_dataset = DFGDataset(dataset)
    
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Advanced Training for DFG Classifier")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_path', type=str, default='dfg-classifier/data/processed', help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='dfg-classifier/models/checkpoints', help='Output directory')
    parser.add_argument('--dfg_mapping', type=str, default='data/dfg_mapping.json', help='Path to DFG mapping')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO', log_file=os.path.join(args.output_dir, 'training.log'))
    
    logger.info("=" * 80)
    logger.info("DFG CLASSIFIER - ADVANCED TRAINING")
    logger.info("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Load DFG mapping
    logger.info("Loading DFG mapping...")
    dfg_mapping = load_dfg_mapping(args.dfg_mapping)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    
    logger.info("Loading training data...")
    train_dataset = load_processed_dataset(args.data_path, 'train')
    logger.info(f"Loaded {len(train_dataset)} training samples")
    
    logger.info("Loading validation data...")
    val_dataset = load_processed_dataset(args.data_path, 'val')
    logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    try:
        logger.info("Loading test data...")
        test_dataset = load_processed_dataset(args.data_path, 'test')
        logger.info(f"Loaded {len(test_dataset)} test samples")
    except FileNotFoundError:
        logger.warning("Test dataset not found, skipping...")
        test_dataset = None
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config.get('device', {}).get('num_workers', 0)
    
    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_data_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_data_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers) if test_dataset else None
    
    logger.info(f"Created data loaders with batch_size={batch_size}")
    
    # Initialize model
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 80)
    
    model_config = config.get('model', {})
    
    model = DFGClassifier(
        model_name=model_config.get('name', 'allenai/scibert_scivocab_uncased'),
        num_classes=model_config.get('num_classes', 30),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        freeze_bert=model_config.get('freeze_bert', False),
        hierarchical=model_config.get('hierarchical', False),
        dfg_mapping=dfg_mapping
    )
    
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"Number of classes: {model_info['num_classes']}")
    
    # Initialize advanced trainer
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING TRAINER")
    logger.info("=" * 80)
    
    trainer = AdvancedTrainer(
        model=model,
        config=config,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Save config
    config_save_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved training configuration to {config_save_path}")
    
    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    num_epochs = config['training']['num_epochs']
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        output_dir=args.output_dir,
        save_best_only=True
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nModel checkpoints saved to: {args.output_dir}")
    logger.info(f"Best validation F1 score: {trainer.best_val_f1:.4f}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total epochs trained: {trainer.epoch + 1}")
    logger.info(f"Total training steps: {trainer.global_step}")
    logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")
    logger.info(f"Final training loss: {trainer.history['train_loss'][-1]:.4f}")
    logger.info(f"Final validation loss: {trainer.history['val_loss'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {trainer.history['val_accuracy'][-1]:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Review training history in training_history.json")
    logger.info("2. Evaluate the model using the best checkpoint")
    logger.info("3. Use the model for inference with src/classify.py")


if __name__ == "__main__":
    main()




