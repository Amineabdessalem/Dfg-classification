# DFG Classifier - Fixed Project Structure

## 🔒 FIXED COMPONENTS (DO NOT MODIFY)

### Core Code (Stable Algorithm & Logic)
```
src/
├── train.py              # Training logic (Trainer class)
├── model.py              # Model architecture (SciBERT classifier)
├── data_processor.py     # Data processing pipeline
├── classify.py           # Inference/prediction
└── utils.py              # Helper functions
```

### Training Runner
```
run_training.py           # Simple script to start training
```

**These files contain the fixed algorithm and should NOT be changed.**

---

## ✏️ MODIFIABLE COMPONENTS (User Configuration)

### 1. Configuration File
```
config.yaml               # Training parameters & DFG codes
```

**What you can change:**
- `model.allowed_labels`: Add/remove DFG codes (e.g., ["4.41", "4.42", "4.43"])
- `model.num_classes`: Update to match number of labels
- `training.num_epochs`: Adjust training duration
- `training.batch_size`: Adjust for your GPU
- Other hyperparameters

### 2. Dataset
```
data/raw/gepris/all_english_data.json    # Your training data
```

**Format (required):**
```json
[
  {
    "title": "Project title",
    "abstract": "Project abstract text",
    "dfg_label": "4.41",
    "project_id": "unique_id",
    "source_url": "https://..."
  }
]
```

**To add new data:**
1. Add entries to `all_english_data.json`
2. Ensure `dfg_label` matches codes in `config.yaml`
3. Run: `python run_training.py`

---

## 📊 OUTPUT STRUCTURE (Auto-Generated)

### Training Outputs
```
models/trained_model/
├── best_model.pth              # Best model checkpoint
├── config.json                 # Training configuration snapshot
├── training_history.json       # 📈 EVALUATION METRICS
└── label_mapping.json          # Label mappings
```

### Training History Format
```json
{
  "train_loss": [1.2, 0.8, 0.5, ...],
  "val_loss": [1.1, 0.9, 0.6, ...],
  "val_accuracy": [0.65, 0.78, 0.85, ...],
  "val_f1": [0.60, 0.75, 0.82, ...],
  "val_precision": [0.62, 0.76, 0.83, ...],
  "val_recall": [0.58, 0.74, 0.81, ...]
}
```

### Logs
```
logs/training.log            # Detailed training logs
```

---

## 🚀 WORKFLOW (Fixed Process)

### Step 1: Prepare Data
- Add/update entries in `data/raw/gepris/all_english_data.json`
- Ensure all entries have: `title`, `abstract`, `dfg_label`

### Step 2: Update Configuration
- Edit `config.yaml`:
  ```yaml
  model:
    allowed_labels: ["4.41", "4.42", "4.43"]  # Your DFG codes
    num_classes: 3                             # Count of labels
  ```

### Step 3: Train
```bash
python run_training.py
```

### Step 4: Check Results
- **Metrics**: `models/trained_model/training_history.json`
- **Logs**: `logs/training.log`
- **Best epoch**: Automatically saved based on validation F1

### Step 5: Use Model
```bash
python src/classify.py --model models/trained_model --input your_document.pdf
```

---

## 📁 Complete Directory Structure

```
Dfg-classification/
│
├── 🔒 FIXED (Do not modify)
│   ├── src/
│   │   ├── train.py
│   │   ├── model.py
│   │   ├── data_processor.py
│   │   ├── classify.py
│   │   └── utils.py
│   ├── run_training.py
│   └── data/dfg_mapping.json
│
├── ✏️ MODIFIABLE (User configuration)
│   ├── config.yaml
│   └── data/raw/gepris/all_english_data.json
│
└── 📊 AUTO-GENERATED (Training outputs)
    ├── models/trained_model/
    │   ├── best_model.pth
    │   ├── training_history.json  ← YOUR EVALUATION RESULTS
    │   └── config.json
    ├── data/processed/
    └── logs/training.log
```

---

## 🎯 Key Principles

1. **Fixed Algorithm**: Core logic in `src/` never changes
2. **User Control**: Only modify `config.yaml` and data files
3. **Automatic Evaluation**: Results saved to `training_history.json`
4. **Reproducible**: Same data + same config = same results

---

## 📈 Evaluation Metrics Explained

After training, check `models/trained_model/training_history.json`:

- **val_f1**: F1-score on validation set (main metric)
- **val_accuracy**: Classification accuracy
- **val_precision**: Precision score
- **val_recall**: Recall score
- **train_loss**: Training loss per epoch
- **val_loss**: Validation loss per epoch

**Best model** is automatically selected based on highest `val_f1`.

---

## 🔄 To Retrain with New Data

1. Add new entries to `data/raw/gepris/all_english_data.json`
2. Update `config.yaml` if adding new DFG codes
3. Run: `python run_training.py`
4. Check: `models/trained_model/training_history.json`

**That's it! No code changes needed.**

---

*This structure is designed to be stable and unchanging. Only modify configuration and data files.*

