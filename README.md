# DFG Subject Area Classifier

Automated classification of research projects into DFG (Deutsche Forschungsgemeinschaft) subject areas using SciBERT.

## 🎯 Current Configuration

**Classes:** 3-class classification
- 4.41: Systems Engineering
- 4.42: Electrical Engineering and Information Technology  
- 4.43: Computer Science

**Dataset:** 191 English projects
- 100 Systems Engineering (synthetic)
- 46 Electrical Engineering (real GEPRIS)
- 45 Computer Science (real GEPRIS)

## 🚀 Quick Start

### Train Model
```bash
python run_training.py
```

### Check Results
```bash
# View training history
cat models/trained_model/training_history.json

# View logs
cat logs/training.log
```

### Use Model for Prediction
```bash
python src/classify.py --model models/trained_model --input your_document.pdf
```

## 📁 Project Structure

```
Dfg-classification/
├── config.yaml                    # Training configuration
├── run_training.py                # Training script
├── data/
│   ├── raw/gepris/
│   │   └── all_english_data.json  # Training data (191 projects)
│   └── processed/all_english/
│       └── processed_dataset.json # Tokenized data
├── src/                           # Core code (DO NOT MODIFY)
│   ├── train.py
│   ├── model.py
│   ├── data_processor.py
│   └── classify.py
├── models/trained_model/          # Training outputs
│   ├── best_model.pth
│   ├── training_history.json      # ← Evaluation metrics
│   └── config.json
└── logs/training.log
```

## ⚙️ Configuration

Edit `config.yaml` to:
- Add/remove DFG codes (`model.allowed_labels`)
- Adjust training parameters
- Change batch size for your GPU

## 📊 Adding New Data

1. Add entries to `data/raw/gepris/all_english_data.json`:
```json
{
  "title": "Project title",
  "abstract": "Project abstract",
  "dfg_label": "4.41",
  "project_id": "unique_id"
}
```

2. Update `config.yaml` if adding new DFG codes:
```yaml
model:
  allowed_labels: ["4.41", "4.42", "4.43", "4.44"]  # Add new code
  num_classes: 4  # Update count
```

3. Train:
```bash
python run_training.py
```

## 📈 Evaluation Metrics

After training, check `models/trained_model/training_history.json`:

```json
{
  "train_loss": [0.90, 0.45, 0.23, ...],
  "val_loss": [0.26, 0.18, 0.15, ...],
  "val_accuracy": [0.95, 0.97, 0.98, ...],
  "val_f1": [0.95, 0.96, 0.97, ...],
  "val_precision": [...],
  "val_recall": [...]
}
```

**Best model** is automatically selected based on highest validation F1 score.

## 🔧 Technical Details

- **Model:** SciBERT (`allenai/scibert_scivocab_uncased`)
- **Max Sequence Length:** 128 tokens (optimized for 4GB GPU)
- **Batch Size:** 2 (with gradient accumulation = 8, effective batch size = 16)
- **Optimizer:** AdamW
- **Learning Rate:** 2e-5
- **Early Stopping:** Patience = 5 epochs

## 📝 Workflow

1. **Prepare data** → `data/raw/gepris/all_english_data.json`
2. **Configure** → `config.yaml`
3. **Train** → `python run_training.py`
4. **Evaluate** → `models/trained_model/training_history.json`
5. **Use** → `python src/classify.py`

## 🎓 Model Performance

Current model (3-class):
- **Validation F1:** 0.95+
- **Validation Accuracy:** 95%+
- **Training samples:** 152
- **Validation samples:** 19
- **Test samples:** 20

## 📚 References

- [DFG Classification System](https://www.dfg.de/en/research_funding/programmes/list/index.jsp)
- [GEPRIS Database](https://gepris.dfg.de/)
- [SciBERT Paper](https://arxiv.org/abs/1903.10676)

---

**Note:** Core algorithm in `src/` is stable and should not be modified. Only update `config.yaml` and data files for new experiments.
