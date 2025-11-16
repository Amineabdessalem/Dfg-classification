# DFG Subject Area Classifier

A machine learning system for classifying scientific/research papers into DFG (German Research Foundation) subject areas using fine-tuned SciBERT. This system supports multi-level hierarchical classification across 4 levels with ~218 review boards.

## 🎯 Project Overview

The DFG Subject Area Classifier is designed to automatically categorize scientific papers into the German Research Foundation's classification system. It uses state-of-the-art transformer models (SciBERT) fine-tuned for scientific text classification.

### Classification Structure

The system supports 4 hierarchical levels:

- **Level 1**: 4 main areas (Humanities & Social Sciences, Life Sciences, Natural Sciences, Engineering Sciences)
- **Level 2**: ~30 subject areas (e.g., Ancient Cultures, Basic Research in Biology, Molecular Chemistry)
- **Level 3**: Research areas (e.g., Prehistory and World Archaeology, Greek and Latin Philology)
- **Level 4**: ~218 review boards (e.g., 1.11-01=Prehistory and World Archaeology)

## 🚀 Features

- **SciBERT-based Architecture**: Uses `allenai/scibert_scivocab_uncased` for scientific text understanding
- **Multi-level Classification**: Supports both single-level and hierarchical classification
- **PDF Processing**: Automatic text extraction from PDF papers
- **Title & Abstract Focus**: Optimized for title and abstract classification
- **Command-line Interface**: Easy-to-use CLI for inference
- **Batch Processing**: Support for classifying multiple papers at once
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## 📁 Project Structure

```
dfg-classifier/
├── data/
│   ├── raw/              # Raw PDF papers
│   ├── processed/        # Processed datasets
│   └── dfg_mapping.json  # DFG code to name mapping
├── models/
│   ├── checkpoints/      # Saved model checkpoints
│   └── scibert/          # Base model cache
├── src/
│   ├── data_processor.py # PDF processing and data preparation
│   ├── model.py         # Model architecture definitions
│   ├── train.py         # Training pipeline
│   ├── classify.py      # Inference CLI
│   └── utils.py         # Utility functions
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── config.yaml         # Configuration file
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dfg-classifier
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SciBERT model** (optional, will be downloaded automatically):
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'); AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')"
   ```

## 📊 Data Preparation

### 1. Prepare Your Data

Create a JSON file with paper filenames and their DFG labels:

```json
{
  "paper_001.pdf": "2.11",
  "paper_002.pdf": "3.12",
  "paper_003.pdf": "1.13"
}
```

### 2. Place PDFs in Data Directory

```
data/raw/
├── paper_001.pdf
├── paper_002.pdf
└── paper_003.pdf
```

### 3. Process the Data

```python
from src.data_processor import DFGDatasetProcessor, load_config, load_dfg_mapping

# Load configuration and mapping
config = load_config("config.yaml")
dfg_mapping = load_dfg_mapping("data/dfg_mapping.json")

# Initialize processor
processor = DFGDatasetProcessor(config, dfg_mapping)

# Create dataset
df = processor.create_dataset_from_pdfs("data/raw/", "data/processed/labels.json")

# Prepare for training
dataset = processor.prepare_dataset(df)
train_data, val_data, test_data = processor.split_dataset(dataset)

# Save processed dataset
processor.save_processed_dataset(train_data, "data/processed/train.json")
processor.save_processed_dataset(val_data, "data/processed/val.json")
processor.save_processed_dataset(test_data, "data/processed/test.json")
```

## 🏋️ Training

### Basic Training

```bash
python src/train.py --data_path data/processed --output_dir models/checkpoints
```

### Advanced Training Options

```bash
python src/train.py \
    --config config.yaml \
    --data_path data/processed \
    --output_dir models/checkpoints \
    --dfg_mapping data/dfg_mapping.json
```

### Training Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  name: "allenai/scibert_scivocab_uncased"
  num_classes: 30
  dropout_rate: 0.3
  hierarchical: false

training:
  learning_rate: 2e-5
  batch_size: 8
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
```

## 🔍 Inference

### Classify a Single PDF

```bash
python src/classify.py --paper data/raw/paper.pdf --model models/checkpoints/best_model.pt
```

### Classify Text Directly

```bash
python src/classify.py \
    --text "This paper presents a novel machine learning approach..." \
    --title "Machine Learning in Scientific Research" \
    --model models/checkpoints/best_model.pt
```

### Batch Classification

```bash
python src/classify.py \
    --batch data/raw/*.pdf \
    --model models/checkpoints/best_model.pt \
    --output results.json
```

### Advanced Inference Options

```bash
python src/classify.py \
    --paper paper.pdf \
    --model models/checkpoints/best_model.pt \
    --top-k 3 \
    --format json \
    --output predictions.json
```

## 📈 Evaluation

### Training Metrics

The training pipeline automatically tracks:
- Training and validation loss
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Learning curves

### Evaluation Reports

```python
from src.utils import create_evaluation_report

# Generate comprehensive evaluation report
create_evaluation_report(
    y_true=true_labels,
    y_pred=predicted_labels,
    class_names=class_names,
    output_path="evaluation_report.json"
)
```

### Visualization

```python
from src.utils import plot_training_history, plot_confusion_matrix

# Plot training history
plot_training_history(history, "training_plots.png")

# Plot confusion matrix
plot_confusion_matrix(confusion_matrix, class_names, "confusion_matrix.png")
```

## ⚙️ Configuration

### Model Configuration

```yaml
model:
  name: "allenai/scibert_scivocab_uncased"  # Base model
  max_sequence_length: 512                   # Max input length
  num_classes: 30                           # Number of classes
  dropout_rate: 0.3                         # Dropout rate
  freeze_bert: false                        # Freeze BERT weights
  hierarchical: false                       # Enable hierarchical mode
```

### Training Configuration

```yaml
training:
  learning_rate: 2e-5                       # Learning rate
  batch_size: 8                            # Batch size
  num_epochs: 3                            # Number of epochs
  warmup_steps: 100                        # Warmup steps
  weight_decay: 0.01                       # Weight decay
  early_stopping_patience: 3               # Early stopping patience
```

### Data Configuration

```yaml
data:
  train_split: 0.8                         # Training set ratio
  val_split: 0.1                           # Validation set ratio
  test_split: 0.1                          # Test set ratio
  random_seed: 42                          # Random seed
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_data_processor.py -v
python -m pytest tests/test_model.py -v
```

## 📊 Performance

### Expected Performance

- **Accuracy**: 85-95% (depending on dataset quality and size)
- **Training Time**: 2-4 hours (on GPU with ~1000 papers)
- **Inference Speed**: ~100 papers/minute (on GPU)

### Model Sizes

- **SciBERT Base**: ~110M parameters
- **Classification Head**: ~9K parameters (for 30 classes)
- **Total Model Size**: ~420MB

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SciBERT](https://github.com/allenai/scibert) by Allen Institute for AI
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- German Research Foundation (DFG) for the classification system

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [link-to-docs]

## 🔄 Version History

- **v1.0.0** - Initial release with SciBERT-based classification
- **v0.9.0** - Beta version with basic functionality
- **v0.8.0** - Alpha version for testing

---

**Note**: This is a research prototype. For production use, additional validation, testing, and optimization may be required.

