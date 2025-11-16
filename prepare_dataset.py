"""
Comprehensive Dataset Preparation Pipeline for DFG Classifier
This script prepares high-quality training data using multiple strategies:
1. Synthetic data generation
2. Data augmentation
3. Class balancing
4. Train/val/test splitting
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processor import DFGDatasetProcessor, load_config, load_dfg_mapping
from synthetic_data_generator import SyntheticPaperGenerator, create_training_ready_dataset
from data_augmentation import TextAugmenter, DatasetBalancer, create_augmented_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_synthetic_dataset(
    dfg_mapping: Dict,
    config: Dict,
    samples_per_category: int = 100,
    use_augmentation: bool = True,
    augmentation_factor: int = 2,
    balance_classes: bool = True,
    output_dir: str = 'dfg-classifier/data/processed'
) -> None:
    """
    Prepare synthetic dataset for training
    
    Args:
        dfg_mapping: DFG mapping dictionary
        config: Configuration dictionary
        samples_per_category: Number of synthetic samples per category
        use_augmentation: Whether to use data augmentation
        augmentation_factor: Augmentation factor (samples per original)
        balance_classes: Whether to balance classes
        output_dir: Output directory for processed data
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Generating Synthetic Dataset")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic papers
    generator = SyntheticPaperGenerator(dfg_mapping)
    synthetic_papers = generator.generate_dataset(
        samples_per_category=samples_per_category,
        output_dir=None  # Don't save intermediate results
    )
    
    logger.info(f"Generated {len(synthetic_papers)} synthetic papers")
    
    # Initialize data processor
    processor = DFGDatasetProcessor(config, dfg_mapping)
    
    # Convert to dataset format
    dataset = []
    for paper in synthetic_papers:
        # Combine title and abstract
        combined_text = f"{paper['title']} [SEP] {paper['abstract']}"
        
        # Tokenize
        tokenized = processor.tokenize_text(combined_text)
        
        # Get label ID
        label_id = processor.label_to_id.get(paper['category'], -1)
        
        if label_id == -1:
            logger.warning(f"Unknown category: {paper['category']}")
            continue
        
        dataset.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': label_id,
            'filename': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'label': paper['category'],
            'combined_text': combined_text
        })
    
    logger.info(f"Created {len(dataset)} training samples")
    
    # Data Augmentation
    if use_augmentation:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Applying Data Augmentation")
        logger.info("=" * 80)
        
        augmenter = TextAugmenter(
            model_name=config.get('model', {}).get('name', 'allenai/scibert_scivocab_uncased'),
            augmentation_prob=0.3
        )
        
        dataset = create_augmented_dataset(
            dataset,
            augmenter,
            augmentation_factor=augmentation_factor,
            balance_classes=False  # We'll balance separately
        )
        
        logger.info(f"Augmented dataset size: {len(dataset)}")
    
    # Class Balancing
    if balance_classes:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Balancing Classes")
        logger.info("=" * 80)
        
        balancer = DatasetBalancer(strategy='oversample')
        dataset = balancer.balance_dataset(dataset)
        
        logger.info(f"Balanced dataset size: {len(dataset)}")
    
    # Split dataset
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Splitting Dataset")
    logger.info("=" * 80)
    
    train_data, val_data, test_data = processor.split_dataset(dataset)
    
    logger.info(f"Train set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")
    
    # Save datasets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving Processed Datasets")
    logger.info("=" * 80)
    
    processor.save_processed_dataset(train_data, os.path.join(output_dir, 'train.json'))
    processor.save_processed_dataset(val_data, os.path.join(output_dir, 'val.json'))
    processor.save_processed_dataset(test_data, os.path.join(output_dir, 'test.json'))
    
    # Save statistics
    stats = {
        'total_samples': len(dataset),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'num_classes': len(processor.label_to_id),
        'samples_per_category': samples_per_category,
        'augmentation_factor': augmentation_factor if use_augmentation else 1,
        'balanced': balance_classes
    }
    
    # Calculate class distribution
    from collections import Counter
    train_labels = [sample['label'] for sample in train_data]
    class_distribution = Counter(train_labels)
    stats['class_distribution'] = dict(class_distribution)
    
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved dataset statistics to {stats_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DATASET PREPARATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nDatasets saved to: {output_dir}")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Ready for training!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare high-quality dataset for DFG classifier training"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dfg-mapping',
        type=str,
        default='data/dfg_mapping.json',
        help='Path to DFG mapping file'
    )
    
    parser.add_argument(
        '--samples-per-category',
        type=int,
        default=100,
        help='Number of synthetic samples to generate per category'
    )
    
    parser.add_argument(
        '--augmentation-factor',
        type=int,
        default=2,
        help='Data augmentation factor (1 = no augmentation)'
    )
    
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--no-balancing',
        action='store_true',
        help='Disable class balancing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dfg-classifier/data/processed',
        help='Output directory for processed datasets'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Load DFG mapping
    logger.info("Loading DFG mapping...")
    dfg_mapping = load_dfg_mapping(args.dfg_mapping)
    
    # Prepare dataset
    prepare_synthetic_dataset(
        dfg_mapping=dfg_mapping,
        config=config,
        samples_per_category=args.samples_per_category,
        use_augmentation=not args.no_augmentation,
        augmentation_factor=args.augmentation_factor,
        balance_classes=not args.no_balancing,
        output_dir=args.output_dir
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("1. Review the generated dataset statistics")
    logger.info("2. Run training with:")
    logger.info(f"   python src/train.py --data_path {args.output_dir}")
    logger.info("3. Or use advanced trainer with:")
    logger.info(f"   python train_advanced.py --data_path {args.output_dir}")


if __name__ == "__main__":
    main()




