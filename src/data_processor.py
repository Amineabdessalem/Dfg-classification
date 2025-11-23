"""
Data processing module for DFG Subject Area Classifier
Handles PDF text extraction, preprocessing, and tokenization for SciBERT
"""

import os
import re
import json
import logging
import yaml
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer
from PyPDF2 import PdfReader
import pdfplumber
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model', {}).get('name', 'allenai/scibert_scivocab_uncased')
        )
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pdfplumber' or 'pypdf2')
            
        Returns:
            Extracted text as string
        """
        try:
            if method == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            elif method == "pypdf2":
                return self._extract_with_pypdf2(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (faster but less accurate)"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_title_and_abstract(self, text: str) -> Tuple[str, str]:
        """
        Extract title and abstract from paper text
        
        Args:
            text: Full text of the paper
            
        Returns:
            Tuple of (title, abstract)
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Try to find title (usually at the beginning, before abstract)
        title = self._extract_title(text)
        
        # Try to find abstract
        abstract = self._extract_abstract(text)
        
        return title, abstract
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def _extract_title(self, text: str) -> str:
        """Extract title from paper text"""
        lines = text.split('\n')
        
        # Look for title patterns
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                # Skip common non-title patterns
                if not any(pattern in line.lower() for pattern in [
                    'abstract', 'introduction', 'keywords', 'author', 'university'
                ]):
                    return line
        
        # Fallback: return first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                return line
        
        return "No title found"
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from paper text"""
        # Look for abstract section
        abstract_patterns = [
            r'abstract[:\s]+(.*?)(?=introduction|keywords|1\.|\n\n)',
            r'abstract[:\s]+(.*?)(?=\n\n)',
            r'abstract\s+(.*?)(?=introduction|keywords)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Ensure it's substantial
                    return abstract
        
        # Fallback: look for text before introduction
        intro_match = re.search(r'introduction', text, re.IGNORECASE)
        if intro_match:
            potential_abstract = text[:intro_match.start()].strip()
            # Take the last substantial paragraph before introduction
            paragraphs = potential_abstract.split('\n\n')
            for para in reversed(paragraphs):
                para = para.strip()
                if len(para) > 100:  # Substantial paragraph
                    return para
        
        return "No abstract found"


class DFGDatasetProcessor:
    """Handles DFG dataset processing and tokenization"""
    
    def __init__(self, config: Dict, dfg_mapping: Dict = None):
        self.config = config
        self.dfg_mapping = dfg_mapping or {}
        self.allowed_labels = (
            self.config.get('model', {}).get('allowed_labels')
            if self.config else None
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model', {}).get('name', 'allenai/scibert_scivocab_uncased')
        )
        
        # Load DFG labels from mapping
        self._load_dfg_labels()
    
    def _load_dfg_labels(self):
        """Load DFG labels from mapping file"""
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
                self.id_to_name = {idx: level_2_classes[code] for code, idx in self.label_to_id.items()}
            else:
                # Fallback to default labels
                self._create_default_labels()
        except Exception as e:
            logger.warning(f"Could not load DFG mapping: {e}. Using default labels.")
            self._create_default_labels()
    
    def _create_default_labels(self):
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
        
        self.label_to_id = {code: idx for idx, code in enumerate(default_labels.keys())}
        self.id_to_label = {idx: code for code, idx in self.label_to_id.items()}
        self.id_to_name = {idx: name for code, idx in self.label_to_id.items() for name in [default_labels[code]]}
    
    def create_dataset_from_pdfs(self, pdf_dir: str, labels_file: str) -> pd.DataFrame:
        """
        Create dataset from PDF files and labels
        
        Args:
            pdf_dir: Directory containing PDF files
            labels_file: JSON file with PDF filenames and their labels
            
        Returns:
            DataFrame with columns: filename, text, title, abstract, label, label_id
        """
        processor = PDFProcessor(self.config)
        
        # Load labels
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        dataset = []
        
        for filename, label in labels_data.items():
            pdf_path = os.path.join(pdf_dir, filename)
            
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
            
            # Extract text
            text = processor.extract_text_from_pdf(pdf_path)
            title, abstract = processor.extract_title_and_abstract(text)
            
            # Combine title and abstract for classification
            combined_text = f"{title} [SEP] {abstract}"
            
            dataset.append({
                'filename': filename,
                'text': text,
                'title': title,
                'abstract': abstract,
                'combined_text': combined_text,
                'label': label,
                'label_id': self.label_to_id.get(label, -1)
            })
        
        return pd.DataFrame(dataset)
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text for SciBERT
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary with tokenized inputs
        """
        max_length = self.config.get('model', {}).get('max_sequence_length', 512)
        
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
    def prepare_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare dataset for training
        
        Args:
            df: DataFrame with text and labels
            
        Returns:
            List of dictionaries with tokenized data
        """
        dataset = []
        
        for _, row in df.iterrows():
            if row['label_id'] == -1:  # Skip invalid labels
                continue
            
            tokenized = self.tokenize_text(row['combined_text'])
            
            dataset.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': torch.tensor(row['label_id'], dtype=torch.long),
                'filename': row['filename'],
                'title': row['title'],
                'abstract': row['abstract'],
                'label': row['label']
            })
        
        return dataset
    
    def split_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            dataset: List of data samples
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        data_config = self.config.get('data', {})
        test_size = data_config.get('test_split', 0.1)
        val_size = data_config.get('val_split', 0.1)
        random_seed = data_config.get('random_seed', 42)
        
        # Extract labels for stratification
        def get_label(item):
            label = item['labels']
            # Handle both tensors and integers
            if isinstance(label, torch.Tensor):
                return label.item()
            return label
        
        labels = [get_label(item) for item in dataset]
        
        # First split: separate test set
        train_val, test = train_test_split(
            dataset, 
            test_size=test_size, 
            random_state=random_seed,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        train_val_labels = [get_label(item) for item in train_val]
        
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=train_val_labels
        )
        
        return train, val, test
    
    def save_processed_dataset(self, dataset: List[Dict], output_path: str):
        """Save processed dataset to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_data = []
        for item in dataset:
            # Handle both tensors and plain Python types
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            serializable_item = {
                'input_ids': input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids,
                'attention_mask': attention_mask.tolist() if isinstance(attention_mask, torch.Tensor) else attention_mask,
                'labels': labels.item() if isinstance(labels, torch.Tensor) else labels,
                'filename': item['filename'],
                'title': item['title'],
                'abstract': item['abstract'],
                'label': item['label']
            }
            serializable_data.append(serializable_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed dataset to {output_path}")


class DFGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for DFG classification"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dfg_mapping(mapping_path: str = "data/dfg_mapping.json") -> Dict:
    """Load DFG mapping from JSON file"""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_labels_file(output_path: str, num_samples: int = 100):
    """
    Create a sample labels file for testing
    
    Args:
        output_path: Path to save the labels file
        num_samples: Number of sample entries to create
    """
    import random
    
    # Load DFG mapping to get valid labels
    try:
        dfg_mapping = load_dfg_mapping()
        level_2_classes = dfg_mapping.get('level_2', {}).get('classes', {})
        label_names = list(level_2_classes.keys())
    except:
        # Fallback to default labels
        label_names = [
            "1.11", "1.12", "1.13", "1.14", "1.15", "1.16",
            "2.11", "2.12", "2.13", "2.14", "2.15", "2.16", "2.17", "2.18",
            "3.11", "3.12", "3.13", "3.14", "3.15", "3.16",
            "4.11", "4.12", "4.13", "4.14", "4.15", "4.16", "4.17", "4.18", "4.19", "4.20"
        ]
    
    sample_labels = {}
    for i in range(num_samples):
        filename = f"paper_{i+1:03d}.pdf"
        label = random.choice(label_names)
        sample_labels[filename] = label
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_labels, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created sample labels file with {num_samples} entries at {output_path}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    dfg_mapping = load_dfg_mapping()
    
    processor = DFGDatasetProcessor(config, dfg_mapping)
    
    # Create sample labels file for testing
    create_sample_labels_file("data/processed/sample_labels.json", 50)
    
    print(f"Loaded {len(processor.label_to_id)} DFG labels")
    print(f"Sample labels: {list(processor.label_to_id.keys())[:5]}")

