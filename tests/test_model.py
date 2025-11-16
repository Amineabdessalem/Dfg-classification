"""
Tests for model architecture
"""

import pytest
import torch
import tempfile
import os

from src.model import SciBERTClassifier, HierarchicalSciBERTClassifier, DFGClassifier, save_model, load_model


class TestSciBERTClassifier:
    """Test SciBERTClassifier class"""
    
    def test_init(self):
        """Test model initialization"""
        model = SciBERTClassifier(num_classes=10)
        
        assert model.num_classes == 10
        assert model.bert is not None
        assert model.classifier is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = SciBERTClassifier(num_classes=5)
        
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 5, (batch_size,))
        
        outputs = model(input_ids, attention_mask, labels)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert 'predictions' in outputs
        assert 'probabilities' in outputs
        
        assert outputs['logits'].shape == (batch_size, 5)
        assert outputs['predictions'].shape == (batch_size,)
        assert outputs['probabilities'].shape == (batch_size, 5)
        assert outputs['loss'] is not None
    
    def test_forward_pass_no_labels(self):
        """Test forward pass without labels"""
        model = SciBERTClassifier(num_classes=5)
        
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'logits' in outputs
        assert outputs['loss'] is None
        assert 'predictions' in outputs
        assert 'probabilities' in outputs


class TestHierarchicalSciBERTClassifier:
    """Test HierarchicalSciBERTClassifier class"""
    
    def test_init(self):
        """Test hierarchical model initialization"""
        num_classes_per_level = {1: 4, 2: 10, 3: 20}
        model = HierarchicalSciBERTClassifier(num_classes_per_level=num_classes_per_level)
        
        assert model.num_classes_per_level == num_classes_per_level
        assert len(model.classifiers) == 3
        assert 'level_1' in model.classifiers
        assert 'level_2' in model.classifiers
        assert 'level_3' in model.classifiers
    
    def test_forward_pass(self):
        """Test hierarchical forward pass"""
        num_classes_per_level = {1: 4, 2: 10}
        model = HierarchicalSciBERTClassifier(num_classes_per_level=num_classes_per_level)
        
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        labels = {
            1: torch.randint(0, 4, (batch_size,)),
            2: torch.randint(0, 10, (batch_size,))
        }
        
        outputs = model(input_ids, attention_mask, labels)
        
        assert 'level_1' in outputs
        assert 'level_2' in outputs
        assert 'total_loss' in outputs
        
        assert outputs['level_1']['logits'].shape == (batch_size, 4)
        assert outputs['level_2']['logits'].shape == (batch_size, 10)


class TestDFGClassifier:
    """Test DFGClassifier wrapper class"""
    
    def test_init_single_level(self):
        """Test single-level model initialization"""
        model = DFGClassifier(num_classes=30)
        
        assert not model.hierarchical
        assert model.model is not None
        assert len(model.label_to_id) > 0
        assert len(model.id_to_label) > 0
    
    def test_init_hierarchical(self):
        """Test hierarchical model initialization"""
        num_classes_per_level = {1: 4, 2: 30}
        model = DFGClassifier(hierarchical=True, num_classes_per_level=num_classes_per_level)
        
        assert model.hierarchical
        assert model.model is not None
    
    def test_predict_single_level(self):
        """Test prediction for single-level model"""
        model = DFGClassifier(num_classes=5)
        
        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        prediction = model.predict(input_ids, attention_mask)
        
        assert 'prediction_id' in prediction
        assert 'prediction_code' in prediction
        assert 'prediction_name' in prediction
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        model = DFGClassifier(num_classes=10)
        info = model.get_model_info()
        
        assert 'model_name' in info
        assert 'hierarchical' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'num_classes' in info


class TestModelSavingLoading:
    """Test model saving and loading"""
    
    def test_save_and_load_model(self):
        """Test saving and loading model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save model
            model = DFGClassifier(num_classes=5)
            checkpoint_path = os.path.join(temp_dir, 'test_model.pt')
            
            save_model(model, checkpoint_path, epoch=1, metrics={'accuracy': 0.95})
            
            assert os.path.exists(checkpoint_path)
            
            # Load model
            loaded_model = load_model(checkpoint_path)
            
            assert loaded_model.num_classes == 5
            assert loaded_model.hierarchical == False
    
    def test_save_model_with_optimizer(self):
        """Test saving model with optimizer state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = DFGClassifier(num_classes=5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            
            checkpoint_path = os.path.join(temp_dir, 'test_model_with_optimizer.pt')
            
            save_model(
                model,
                checkpoint_path,
                epoch=2,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={'loss': 0.1, 'accuracy': 0.9}
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            assert 'optimizer_state_dict' in checkpoint
            assert 'scheduler_state_dict' in checkpoint
            assert 'metrics' in checkpoint
            assert checkpoint['epoch'] == 2


if __name__ == "__main__":
    pytest.main([__file__])











