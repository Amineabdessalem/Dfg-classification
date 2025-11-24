"""
Model architecture for DFG Subject Area Classifier
SciBERT-based classification model with hierarchical support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
from data_processor import load_config, load_dfg_mapping
import numpy as np
from typing import Dict, Optional, Tuple, List

import logging

logger = logging.getLogger(__name__)


class SciBERTClassifier(nn.Module):
    """
    SciBERT-based classifier for DFG subject area classification
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes: int = 30,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False
    ):
        super(SciBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load SciBERT model and configuration
        self.config = AutoConfig.from_pretrained(model_name)
        # Prefer loading from Flax checkpoint directly into a fresh PyTorch model
        try:
            flax_path = hf_hub_download(model_name, filename='flax_model.msgpack')
            self.bert = AutoModel.from_config(self.config)
            load_flax_checkpoint_in_pytorch_model(self.bert, flax_path)
        except Exception:
            # Fallback: manual .bin (if available)
            bin_path = hf_hub_download(model_name, filename='pytorch_model.bin')
            self.bert = AutoModel.from_config(self.config)
            state = torch.load(bin_path, map_location='cpu', weights_only=True)
            self.bert.load_state_dict(state, strict=False)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT parameters frozen")
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.classifier.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            labels: Ground truth labels tensor [batch_size] (optional)
            
        Returns:
            Dictionary containing logits, loss (if labels provided), and predictions
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (pooler_output)
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'loss': loss,
            'predictions': predictions,
            'probabilities': probabilities
        }


class HierarchicalSciBERTClassifier(nn.Module):
    """
    Hierarchical SciBERT classifier for multi-level DFG classification
    Supports classification at multiple levels (Level 1-4)
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes_per_level: Dict[int, int] = None,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        use_hierarchical_loss: bool = True
    ):
        super(HierarchicalSciBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.use_hierarchical_loss = use_hierarchical_loss
        
        # Default number of classes per level
        if num_classes_per_level is None:
            num_classes_per_level = {
                1: 4,   # Level 1: 4 main areas
                2: 30,  # Level 2: Subject areas
                3: 100, # Level 3: Research areas (estimated)
                4: 218  # Level 4: Review boards
            }
        
        self.num_classes_per_level = num_classes_per_level
        
        # Load SciBERT model
        self.config = AutoConfig.from_pretrained(model_name)
        # Prefer loading from Flax checkpoint directly into a fresh PyTorch model
        try:
            flax_path = hf_hub_download(model_name, filename='flax_model.msgpack')
            self.bert = AutoModel.from_config(self.config)
            load_flax_checkpoint_in_pytorch_model(self.bert, flax_path)
        except Exception:
            # Fallback: manual .bin (if available)
            bin_path = hf_hub_download(model_name, filename='pytorch_model.bin')
            self.bert = AutoModel.from_config(self.config)
            state = torch.load(bin_path, map_location='cpu', weights_only=True)
            self.bert.load_state_dict(state, strict=False)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT parameters frozen")
        
        # Shared feature extraction
        self.dropout = nn.Dropout(dropout_rate)
        
        # Level-specific classifiers
        self.classifiers = nn.ModuleDict({
            f'level_{level}': nn.Linear(self.config.hidden_size, num_classes)
            for level, num_classes in num_classes_per_level.items()
        })
        
        # Initialize classifier weights
        for classifier in self.classifiers.values():
            nn.init.normal_(classifier.weight, std=0.02)
            nn.init.normal_(classifier.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical model
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            labels: Dictionary of ground truth labels per level [batch_size] (optional)
            
        Returns:
            Dictionary containing logits, losses, and predictions for each level
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Forward through each level classifier
        results = {}
        total_loss = 0
        
        for level in self.num_classes_per_level.keys():
            classifier = self.classifiers[f'level_{level}']
            logits = classifier(pooled_output)
            
            # Calculate loss for this level if labels provided
            level_loss = None
            if labels is not None and level in labels:
                loss_fct = nn.CrossEntropyLoss()
                level_loss = loss_fct(logits, labels[level])
                total_loss += level_loss
            
            # Get predictions and probabilities
            predictions = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)
            
            results[f'level_{level}'] = {
                'logits': logits,
                'loss': level_loss,
                'predictions': predictions,
                'probabilities': probabilities
            }
        
        # Add total loss if hierarchical loss is enabled
        if self.use_hierarchical_loss and labels is not None:
            results['total_loss'] = total_loss
        
        return results


class DFGClassifier(nn.Module):
    """
    Main DFG classifier wrapper with additional utilities
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes: int = 30,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False,
        hierarchical: bool = False,
        num_classes_per_level: Dict[int, int] = None,
        dfg_mapping: Dict = None,
        allowed_labels: Optional[List[str]] = None
    ):
        super(DFGClassifier, self).__init__()
        
        self.hierarchical = hierarchical
        self.dfg_mapping = dfg_mapping or {}
        self.allowed_labels = allowed_labels
        
        if allowed_labels and not hierarchical:
            num_classes = len(allowed_labels)
        if hierarchical:
            self.model = HierarchicalSciBERTClassifier(
                model_name=model_name,
                num_classes_per_level=num_classes_per_level,
                dropout_rate=dropout_rate,
                freeze_bert=freeze_bert
            )
        else:
            self.model = SciBERTClassifier(
                model_name=model_name,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                freeze_bert=freeze_bert
            )
        
        # Load label mappings from DFG mapping
        self._load_label_mappings()
    
    def _load_label_mappings(self):
        """Load label mappings from DFG mapping"""
        try:
            if self.dfg_mapping:
                level_2_classes = self.dfg_mapping.get('level_2', {}).get('classes', {})
                if self.allowed_labels:
                    level_2_classes = {
                        code: name for code, name in level_2_classes.items()
                        if code in self.allowed_labels
                    }
                self.label_to_id = {code: idx for idx, code in enumerate(level_2_classes.keys())}
                self.id_to_label = {idx: code for code, idx in self.label_to_id.items()}
                self.id_to_name = {idx: name for code, idx in self.label_to_id.items() for name in [level_2_classes[code]]}
            else:
                self._create_default_mappings()
        except Exception as e:
            logger.warning(f"Could not load DFG mapping: {e}. Using default mappings.")
            self._create_default_mappings()
    
    def _create_default_mappings(self):
        """Create default label mappings"""
        default_labels = {
            "1.11": "Ancient Cultures", "1.12": "History", "1.13": "Philosophy",
            "1.14": "Theology and Religious Studies", "1.15": "Literature and Linguistics",
            "1.16": "Art and Music Studies", "2.11": "Basic Research in Biology",
            "2.12": "Medicine", "2.13": "Agriculture, Forestry and Veterinary Medicine",
            "2.14": "Psychology", "2.15": "Educational Research", "2.16": "Social Sciences",
            "2.17": "Economics", "2.18": "Law", "3.11": "Mathematics",
            "3.12": "Physics", "3.13": "Chemistry", "3.14": "Geosciences",
            "3.15": "Computer Science", "3.16": "Systems Engineering",
            "4.11": "Mechanical and Industrial Engineering",
            "4.12": "Construction Engineering and Architecture",
            "4.13": "Electrical Engineering and Information Technology",
            "4.14": "Production Technology", "4.15": "Materials Science and Engineering",
            "4.16": "Chemical and Process Engineering", "4.17": "Traffic and Transport Systems",
            "4.18": "Aerospace Engineering", "4.19": "Marine Technology",
            "4.20": "Other Engineering Sciences"
        }
        
        labels = default_labels
        if self.allowed_labels:
            labels = {code: name for code, name in default_labels.items() if code in self.allowed_labels}
        self.label_to_id = {code: idx for idx, code in enumerate(labels.keys())}
        self.id_to_label = {idx: code for code, idx in self.label_to_id.items()}
        self.id_to_name = {idx: labels[code] for code, idx in self.label_to_id.items()}
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.model(*args, **kwargs)
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Make predictions on input data
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            if self.hierarchical:
                # For hierarchical model, return predictions for all levels
                predictions = {}
                for level in outputs.keys():
                    if level.startswith('level_'):
                        pred_id = outputs[level]['predictions'].item()
                        predictions[level] = {
                            'prediction_id': pred_id,
                            'prediction_code': self.id_to_label.get(pred_id, 'Unknown'),
                            'prediction_name': self.id_to_name.get(pred_id, 'Unknown')
                        }
                        if return_probabilities:
                            probs = outputs[level]['probabilities'].cpu().numpy()
                            predictions[level]['probabilities'] = np.asarray(probs).squeeze().tolist()
                return predictions
            else:
                # For single-level model
                pred_id = outputs['predictions'].item()
                result = {
                    'prediction_id': pred_id,
                    'prediction_code': self.id_to_label.get(pred_id, 'Unknown'),
                    'prediction_name': self.id_to_name.get(pred_id, 'Unknown')
                }
                if return_probabilities:
                    probs = outputs['probabilities'].cpu().numpy()
                    result['probabilities'] = np.asarray(probs).squeeze().tolist()
                return result
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.model_name,
            'hierarchical': self.hierarchical,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.model.num_classes if not self.hierarchical else self.model.num_classes_per_level,
            'dropout_rate': self.model.dropout_rate
        }


def _default_model_kwargs() -> Dict:
    """Build default model kwargs from config and mapping."""
    config = load_config("config.yaml")
    dfg_mapping = load_dfg_mapping("data/dfg_mapping.json")
    model_cfg = config.get("model", {})
    allowed_labels = model_cfg.get("allowed_labels")
    num_classes = model_cfg.get("num_classes")
    if not num_classes:
        # fallback to level_2 class count
        level2 = dfg_mapping.get("level_2", {}).get("classes", {})
        if allowed_labels:
            num_classes = len(allowed_labels)
        else:
        num_classes = len(level2)
    elif allowed_labels:
        num_classes = len(allowed_labels)

    return {
        "model_name": model_cfg.get("name", "allenai/scibert_scivocab_uncased"),
        "num_classes": num_classes,
        "dropout_rate": model_cfg.get("dropout_rate", 0.3),
        "freeze_bert": model_cfg.get("freeze_bert", False),
        "hierarchical": model_cfg.get("hierarchical", False),
        "dfg_mapping": dfg_mapping,
        "allowed_labels": allowed_labels,
    }


def load_model(checkpoint_path: str, device: torch.device = None) -> DFGClassifier:
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded DFGClassifier model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine if checkpoint is full dict or raw state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_config = checkpoint.get('model_config') or _default_model_kwargs()
        model = DFGClassifier(**model_config)
        state_dict = checkpoint['model_state_dict']
    else:
        # Raw state dict saved via torch.save(model.state_dict())
        model_config = _default_model_kwargs()
        model = DFGClassifier(**model_config)
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading state dict: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading state dict: {unexpected}")
    model.to(device)
    
    return model


def save_model(
    model: DFGClassifier,
    checkpoint_path: str,
    epoch: int = None,
    optimizer_state_dict: Dict = None,
    scheduler_state_dict: Dict = None,
    metrics: Dict = None
):
    """
    Save model checkpoint
    
    Args:
        model: DFGClassifier model to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        optimizer_state_dict: Optimizer state dictionary
        scheduler_state_dict: Scheduler state dictionary
        metrics: Training metrics dictionary
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model.model.model_name,
            'num_classes': model.model.num_classes if not model.hierarchical else model.model.num_classes_per_level,
            'dropout_rate': model.model.dropout_rate,
            'hierarchical': model.hierarchical,
            'dfg_mapping': model.dfg_mapping,
            'allowed_labels': model.allowed_labels
        }
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_state_dict is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state_dict
    
    if scheduler_state_dict is not None:
        checkpoint['scheduler_state_dict'] = scheduler_state_dict
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    # Test model creation
    print("Testing DFG Classifier model creation...")
    
    # Test single-level model
    model = DFGClassifier(num_classes=30)
    print(f"Single-level model created: {model.get_model_info()}")
    
    # Test hierarchical model
    hierarchical_model = DFGClassifier(
        hierarchical=True,
        num_classes_per_level={1: 4, 2: 30, 3: 100, 4: 218}
    )
    print(f"Hierarchical model created: {hierarchical_model.get_model_info()}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Single-level forward pass
    outputs = model(input_ids, attention_mask)
    print(f"Single-level model outputs keys: {outputs.keys()}")
    
    # Hierarchical forward pass
    hierarchical_outputs = hierarchical_model(input_ids, attention_mask)
    print(f"Hierarchical model outputs keys: {hierarchical_outputs.keys()}")
    
    print("Model tests completed successfully!")

