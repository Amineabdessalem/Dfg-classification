# Jupyter Notebooks for DFG Classifier

This directory contains interactive Jupyter notebooks for experimentation and analysis of the DFG classification project.

## Overview

The notebooks provide an interactive environment for exploring, preparing, training, and evaluating the DFG classifier. Use these for experimentation and learning, while the Python scripts (`train_simple.py`, `train_advanced.py`, `prepare_dataset.py`) are used for production.

## Notebooks

### 01_explore_data.ipynb
**Purpose**: Explore and understand the dataset

**Features**:
- Load and inspect dataset splits (train/val/test)
- View dataset statistics (size, splits, class distribution)
- Visualize class distribution across splits
- Check data quality (empty fields, text lengths)
- Inspect sample data

**Use Cases**:
- Understand dataset structure
- Identify class imbalance issues
- Check data quality before training
- Explore sample data

---

### 02_prepare_dataset.ipynb
**Purpose**: Interactive dataset preparation

**Features**:
- Generate synthetic data with adjustable parameters
- Apply data augmentation with visualization
- Balance classes interactively
- Split dataset into train/val/test
- Visualize augmentation and balancing results
- Adjust parameters and see immediate results

**Use Cases**:
- Experiment with different augmentation strategies
- Try different balancing approaches
- Visualize how augmentation affects class distribution
- Fine-tune dataset preparation parameters

**Key Parameters**:
- `SAMPLES_PER_CATEGORY`: Number of synthetic samples per category
- `USE_AUGMENTATION`: Enable/disable augmentation
- `AUGMENTATION_FACTOR`: Augmentation factor (2 = double dataset)
- `AUGMENTATION_PROB`: Probability of augmenting each sample
- `BALANCE_CLASSES`: Enable/disable class balancing
- `BALANCING_STRATEGY`: 'oversample' or 'undersample'

---

### 03_train_model.ipynb
**Purpose**: Interactive model training

**Features**:
- Adjust hyperparameters interactively
- Monitor training progress in real-time with progress bars
- Visualize training curves (loss, accuracy, F1)
- Try different model configurations
- Save trained models and training history

**Use Cases**:
- Experiment with different hyperparameters
- Visualize training progress
- Understand model behavior during training
- Compare different training configurations

**Key Hyperparameters**:
- `BATCH_SIZE`: Batch size (try 2, 4, 8, 16)
- `NUM_EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate (try 1e-5, 2e-5, 5e-5)
- `DROPOUT_RATE`: Dropout rate (try 0.1, 0.3, 0.5)
- `FREEZE_BERT`: Whether to freeze BERT weights
- `MODEL_NAME`: Pre-trained model name

---

### 04_evaluate_model.ipynb
**Purpose**: Comprehensive model evaluation

**Features**:
- Overall performance metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class performance metrics
- Error analysis (most common errors)
- Sample predictions (correct and incorrect)
- Confidence distribution analysis
- Detailed classification report

**Use Cases**:
- Understand model performance in detail
- Identify problematic classes
- Analyze common misclassifications
- Compare different model versions
- Generate evaluation reports

---

## Getting Started

### Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Jupyter is installed:
```bash
pip install jupyter notebook
```

3. Make sure you have the dataset prepared:
   - Either use existing data in `dfg-classifier/data/processed` or `dfg-classifier/data/small`
   - Or run `prepare_dataset.py` to generate synthetic data

### Running Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory

3. Open notebooks in order:
   - Start with `01_explore_data.ipynb` to understand your data
   - Use `02_prepare_dataset.ipynb` to prepare/refine your dataset
   - Train models interactively with `03_train_model.ipynb`
   - Evaluate with `04_evaluate_model.ipynb`

### Tips

- **Iterate quickly**: Change parameters and re-run cells to see immediate results
- **Visualize everything**: Use the built-in visualizations to understand your data and model
- **Experiment freely**: These notebooks are for exploration - production code uses the scripts
- **Save your work**: Save trained models and important configurations
- **Start small**: Use `dfg-classifier/data/small` for faster experimentation

---

## Workflow

### Recommended Workflow

1. **Exploration** (`01_explore_data.ipynb`)
   - Load and explore existing dataset
   - Understand class distribution
   - Check data quality

2. **Preparation** (`02_prepare_dataset.ipynb`)
   - Generate or refine dataset
   - Experiment with augmentation
   - Balance classes
   - Visualize results

3. **Training** (`03_train_model.ipynb`)
   - Adjust hyperparameters
   - Train model interactively
   - Monitor training curves
   - Save best model

4. **Evaluation** (`04_evaluate_model.ipynb`)
   - Load trained model
   - Evaluate on test set
   - Analyze errors
   - Generate reports

---

## Advantages of Notebooks

✅ **Interactive Experimentation**: Change parameters and see results immediately  
✅ **Visualization**: Live plots, confusion matrices, data analysis  
✅ **Documentation**: Combine code, results, and explanations  
✅ **Learning**: Understand each step interactively  
✅ **Rapid Prototyping**: Quickly test ideas and configurations  

---

## Production vs Experimentation

- **Notebooks** (`notebooks/`): For experimentation, learning, and visualization
- **Scripts** (`train_simple.py`, `train_advanced.py`, `prepare_dataset.py`): For production, automation, and reproducibility

Use notebooks to explore and understand, then use scripts for production training and deployment.

---

## Notes

- All notebooks assume the project structure with `src/`, `data/`, and `dfg-classifier/` directories
- Paths are relative to the notebook directory (`notebooks/`)
- Adjust `DATA_PATH` in each notebook to use different datasets
- Models saved from notebooks can be loaded by evaluation scripts
- Notebooks support both CPU and GPU (automatically detects device)

---

## Troubleshooting

### Import Errors
- Make sure you're running notebooks from the `notebooks/` directory
- Check that `src/` directory exists with all required modules

### File Not Found
- Verify data paths in each notebook
- Run `prepare_dataset.py` if data doesn't exist
- Use `dfg-classifier/data/small` for small test datasets

### Memory Issues
- Reduce `BATCH_SIZE` in training notebook
- Use smaller dataset (`dfg-classifier/data/small`)
- Reduce `SAMPLES_PER_CATEGORY` in dataset preparation

---

## Further Reading

- See `../README.md` for overall project documentation
- See `../QUICKSTART.md` for quick start guide
- Check `../train_simple.py` for production training script

