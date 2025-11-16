"""
Command-line interface for DFG Subject Area Classifier inference
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np

from model import load_model, DFGClassifier
from data_processor import PDFProcessor, DFGDatasetProcessor, load_config, load_dfg_mapping
from utils import setup_logging, format_prediction_output, get_device

# Set up logging
logger = logging.getLogger(__name__)


class DFGClassifierInference:
    """Inference class for DFG classifier"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml", dfg_mapping_path: str = "data/dfg_mapping.json"):
        """
        Initialize inference class
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            dfg_mapping_path: Path to DFG mapping file
        """
        # Load configuration and mapping
        self.config = load_config(config_path)
        self.dfg_mapping = load_dfg_mapping(dfg_mapping_path)
        
        # Set device
        self.device = get_device(self.config.get('device', {}).get('use_cuda', True))
        
        # Load model
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(self.config)
        self.data_processor = DFGDatasetProcessor(self.config, self.dfg_mapping)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model info: {self.model.get_model_info()}")
    
    def classify_pdf(self, pdf_path: str, return_probabilities: bool = True, top_k: int = 5) -> Dict:
        """
        Classify a PDF paper
        
        Args:
            pdf_path: Path to PDF file
            return_probabilities: Whether to return prediction probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Classification results dictionary
        """
        # Extract text from PDF
        logger.info(f"Processing PDF: {pdf_path}")
        text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            return {
                'error': 'Could not extract text from PDF',
                'pdf_path': pdf_path
            }
        
        # Extract title and abstract
        title, abstract = self.pdf_processor.extract_title_and_abstract(text)
        combined_text = f"{title} [SEP] {abstract}"
        
        # Tokenize
        tokenized = self.data_processor.tokenize_text(combined_text)
        
        # Prepare input tensors
        input_ids = tokenized['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = tokenized['attention_mask'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model.predict(input_ids, attention_mask, return_probabilities)
        
        # Format output
        result = {
            'pdf_path': pdf_path,
            'title': title,
            'abstract': abstract,
            'prediction': prediction,
            'processing_info': {
                'text_extracted': len(text) > 0,
                'title_found': title != "No title found",
                'abstract_found': abstract != "No abstract found"
            }
        }
        
        # Add top-k predictions if requested
        if return_probabilities and 'probabilities' in prediction:
            class_names = list(self.data_processor.id_to_label.values())
            formatted_prediction = format_prediction_output(
                prediction, class_names, include_probabilities=True, top_k=top_k
            )
            result['formatted_prediction'] = formatted_prediction
        
        return result
    
    def classify_text(self, text: str, title: str = "", return_probabilities: bool = True, top_k: int = 5) -> Dict:
        """
        Classify text directly
        
        Args:
            text: Input text to classify
            title: Optional title
            return_probabilities: Whether to return prediction probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Classification results dictionary
        """
        # Combine title and text
        if title:
            combined_text = f"{title} [SEP] {text}"
        else:
            combined_text = text
        
        # Tokenize
        tokenized = self.data_processor.tokenize_text(combined_text)
        
        # Prepare input tensors
        input_ids = tokenized['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = tokenized['attention_mask'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model.predict(input_ids, attention_mask, return_probabilities)
        
        # Format output
        result = {
            'input_text': text,
            'title': title,
            'prediction': prediction
        }
        
        # Add top-k predictions if requested
        if return_probabilities and 'probabilities' in prediction:
            class_names = list(self.data_processor.id_to_label.values())
            formatted_prediction = format_prediction_output(
                prediction, class_names, include_probabilities=True, top_k=top_k
            )
            result['formatted_prediction'] = formatted_prediction
        
        return result
    
    def batch_classify_pdfs(self, pdf_paths: List[str], output_path: str = None) -> List[Dict]:
        """
        Classify multiple PDFs in batch
        
        Args:
            pdf_paths: List of PDF file paths
            output_path: Optional path to save results
            
        Returns:
            List of classification results
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.classify_pdf(pdf_path)
                results.append(result)
                logger.info(f"Classified: {pdf_path} -> {result['prediction'].get('prediction_name', 'Unknown')}")
            except Exception as e:
                error_result = {
                    'pdf_path': pdf_path,
                    'error': str(e)
                }
                results.append(error_result)
                logger.error(f"Error classifying {pdf_path}: {e}")
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        return results


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="DFG Subject Area Classifier - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single PDF
  python classify.py --paper paper.pdf --model models/checkpoints/best_model.pt
  
  # Classify text directly
  python classify.py --text "Machine learning algorithms..." --title "ML Paper" --model models/checkpoints/best_model.pt
  
  # Batch classify multiple PDFs
  python classify.py --batch papers/*.pdf --model models/checkpoints/best_model.pt --output results.json
  
  # Classify with top-3 predictions
  python classify.py --paper paper.pdf --model models/checkpoints/best_model.pt --top-k 3
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--paper", type=str, help="Path to PDF paper to classify")
    input_group.add_argument("--text", type=str, help="Text to classify directly")
    input_group.add_argument("--batch", nargs='+', help="List of PDF files for batch classification")
    
    # Model and configuration
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--dfg-mapping", type=str, default="data/dfg_mapping.json", help="Path to DFG mapping file")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file path for results (JSON format)")
    parser.add_argument("--format", type=str, choices=["json", "text"], default="json", help="Output format")
    
    # Classification options
    parser.add_argument("--title", type=str, help="Title for text classification (used with --text)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--no-probabilities", action="store_true", help="Don't include prediction probabilities")
    
    # Logging
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate inputs
    if args.paper and not os.path.exists(args.paper):
        logger.error(f"PDF file not found: {args.paper}")
        return
    
    if args.batch:
        missing_files = [f for f in args.batch if not os.path.exists(f)]
        if missing_files:
            logger.error(f"PDF files not found: {missing_files}")
            return
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    try:
        # Initialize classifier
        classifier = DFGClassifierInference(
            model_path=args.model,
            config_path=args.config,
            dfg_mapping_path=args.dfg_mapping
        )
        
        # Perform classification
        if args.paper:
            # Single PDF classification
            result = classifier.classify_pdf(
                args.paper,
                return_probabilities=not args.no_probabilities,
                top_k=args.top_k
            )
            results = [result]
            
        elif args.text:
            # Text classification
            result = classifier.classify_text(
                args.text,
                title=args.title or "",
                return_probabilities=not args.no_probabilities,
                top_k=args.top_k
            )
            results = [result]
            
        elif args.batch:
            # Batch PDF classification
            results = classifier.batch_classify_pdfs(args.batch, args.output)
        
        # Output results
        if args.format == "json":
            output_data = results[0] if len(results) == 1 else results
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
        
        else:  # text format
            for result in results:
                if 'error' in result:
                    print(f"Error: {result['error']}")
                    continue
                
                prediction = result['prediction']
                print(f"\nClassification Result:")
                print(f"Predicted Class: {prediction.get('prediction_name', 'Unknown')}")
                print(f"Class Code: {prediction.get('prediction_code', 'Unknown')}")
                
                if 'formatted_prediction' in result:
                    formatted = result['formatted_prediction']
                    print(f"Confidence: {formatted['confidence']:.3f}")
                    
                    if 'top_predictions' in formatted:
                        print(f"\nTop {len(formatted['top_predictions'])} Predictions:")
                        for i, pred in enumerate(formatted['top_predictions'], 1):
                            print(f"  {i}. {pred['class_name']} ({pred['probability']:.3f})")
                
                if 'title' in result and result['title']:
                    print(f"\nTitle: {result['title']}")
                
                print("-" * 50)
    
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

