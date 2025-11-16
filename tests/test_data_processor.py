"""
Tests for data processing module
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data_processor import PDFProcessor, DFGDatasetProcessor, DFGDataset


class TestPDFProcessor:
    """Test PDFProcessor class"""
    
    def test_init(self):
        """Test PDFProcessor initialization"""
        processor = PDFProcessor()
        assert processor.tokenizer is not None
    
    def test_extract_title_and_abstract(self):
        """Test title and abstract extraction"""
        processor = PDFProcessor()
        
        # Sample paper text
        sample_text = """
        Title: Machine Learning in Scientific Research
        
        Abstract: This paper presents a novel approach to using machine learning
        algorithms for scientific research. We demonstrate the effectiveness of
        our method on various datasets and show significant improvements.
        
        Introduction: The field of machine learning has seen tremendous growth...
        """
        
        title, abstract = processor.extract_title_and_abstract(sample_text)
        
        assert title != "No title found"
        assert abstract != "No abstract found"
        assert "Machine Learning" in title
        assert "novel approach" in abstract
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        processor = PDFProcessor()
        
        dirty_text = "This    is   a   test    with   extra    spaces."
        clean_text = processor._clean_text(dirty_text)
        
        assert "    " not in clean_text
        assert clean_text == "This is a test with extra spaces."


class TestDFGDatasetProcessor:
    """Test DFGDatasetProcessor class"""
    
    def setup_method(self):
        """Setup test data"""
        self.config = {
            'model': {'name': 'allenai/scibert_scivocab_uncased'},
            'model': {'max_sequence_length': 512},
            'data': {'test_split': 0.2, 'val_split': 0.1, 'random_seed': 42}
        }
        
        self.dfg_mapping = {
            'level_2': {
                'classes': {
                    '1.11': 'Ancient Cultures',
                    '2.11': 'Basic Research in Biology',
                    '3.11': 'Mathematics'
                }
            }
        }
        
        self.processor = DFGDatasetProcessor(self.config, self.dfg_mapping)
    
    def test_init(self):
        """Test DFGDatasetProcessor initialization"""
        assert self.processor.tokenizer is not None
        assert len(self.processor.label_to_id) > 0
        assert len(self.processor.id_to_label) > 0
    
    def test_tokenize_text(self):
        """Test text tokenization"""
        sample_text = "This is a test paper about machine learning."
        tokenized = self.processor.tokenize_text(sample_text)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert tokenized['input_ids'].shape[0] == 512  # max_sequence_length
        assert tokenized['attention_mask'].shape[0] == 512
    
    def test_split_dataset(self):
        """Test dataset splitting"""
        # Create sample dataset
        sample_data = []
        for i in range(100):
            sample_data.append({
                'input_ids': torch.randint(0, 1000, (512,)),
                'attention_mask': torch.ones(512),
                'labels': torch.tensor(i % 3),  # 3 classes
                'filename': f'paper_{i}.pdf',
                'title': f'Paper {i}',
                'abstract': f'Abstract {i}',
                'label': f'class_{i % 3}'
            })
        
        train, val, test = self.processor.split_dataset(sample_data)
        
        # Check splits
        assert len(train) + len(val) + len(test) == 100
        assert len(test) == 20  # 20% of 100
        assert len(val) == 8   # 10% of remaining 80
    
    def test_save_processed_dataset(self):
        """Test saving processed dataset"""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_data = [{
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'labels': torch.tensor(0),
                'filename': 'test.pdf',
                'title': 'Test Title',
                'abstract': 'Test Abstract',
                'label': '1.11'
            }]
            
            output_path = os.path.join(temp_dir, 'test_dataset.json')
            self.processor.save_processed_dataset(sample_data, output_path)
            
            assert os.path.exists(output_path)
            
            # Verify content
            import json
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) == 1
            assert saved_data[0]['filename'] == 'test.pdf'


class TestDFGDataset:
    """Test DFGDataset class"""
    
    def test_init_and_len(self):
        """Test dataset initialization and length"""
        sample_data = [
            {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0)},
            {'input_ids': torch.tensor([4, 5, 6]), 'labels': torch.tensor(1)}
        ]
        
        dataset = DFGDataset(sample_data)
        assert len(dataset) == 2
    
    def test_getitem(self):
        """Test dataset item access"""
        sample_data = [
            {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor(0), 'filename': 'test1.pdf'},
            {'input_ids': torch.tensor([4, 5, 6]), 'labels': torch.tensor(1), 'filename': 'test2.pdf'}
        ]
        
        dataset = DFGDataset(sample_data)
        
        item = dataset[0]
        assert torch.equal(item['input_ids'], torch.tensor([1, 2, 3]))
        assert item['labels'] == 0
        assert item['filename'] == 'test1.pdf'


if __name__ == "__main__":
    pytest.main([__file__])
