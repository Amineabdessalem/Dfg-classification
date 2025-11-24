"""
Simple training runner for DFG Classifier
"""

import sys
from pathlib import Path

sys.path.insert(0, 'src')

from train import Trainer
from data_processor import load_config, load_dfg_mapping
from utils import setup_logging

# Setup
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
setup_logging(log_file='logs/training.log')

# Load config
config = load_config('config.yaml')
dfg_mapping = load_dfg_mapping('data/dfg_mapping.json')

# Paths
data_path = 'data/processed/all_english'  # Directory, not file!
output_dir = 'models/trained_model'

print("="*70)
print("DFG CLASSIFIER TRAINING")
print("="*70)
print(f"\nDataset: {data_path}")
print(f"Output: {output_dir}")
print(f"Classes: {config['model']['allowed_labels']}")
print(f"Epochs: {config['training']['num_epochs']}")
print("="*70 + "\n")

# Train
trainer = Trainer(config, dfg_mapping)
trainer.train(data_path=data_path, output_dir=output_dir)

print("\n✅ Training complete! Model saved to:", output_dir)

