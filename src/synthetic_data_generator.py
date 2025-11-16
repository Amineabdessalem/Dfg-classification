"""
Synthetic Data Generator for DFG Classifier
Generates synthetic scientific paper abstracts for testing and bootstrapping
"""

import random
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SyntheticPaperGenerator:
    """Generate synthetic scientific papers for each DFG category"""
    
    def __init__(self, dfg_mapping: Dict):
        """
        Initialize synthetic paper generator
        
        Args:
            dfg_mapping: DFG mapping dictionary
        """
        self.dfg_mapping = dfg_mapping
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize paper templates for each domain"""
        self.templates = {
            # Computer Science & Engineering
            '3.15': {
                'domain': 'Computer Science',
                'title_templates': [
                    "A Novel {method} for {task} using {technique}",
                    "{technique}-Based Approach to {task}",
                    "Improving {task} with {method}",
                    "Deep Learning Methods for {task}",
                    "{method}: A New Framework for {task}"
                ],
                'methods': ['algorithm', 'neural network', 'framework', 'model', 'architecture'],
                'tasks': ['classification', 'optimization', 'prediction', 'detection', 'segmentation'],
                'techniques': ['deep learning', 'machine learning', 'reinforcement learning', 'transfer learning'],
                'abstract_template': (
                    "This paper proposes a {method} for {task}. "
                    "We introduce a {technique} approach that addresses the limitations of existing methods. "
                    "Our method achieves {improvement}% improvement over baseline approaches. "
                    "Experimental results on {dataset} dataset demonstrate the effectiveness of our approach. "
                    "The proposed method shows promise for {application} applications."
                )
            },
            
            # Physics
            '3.12': {
                'domain': 'Physics',
                'title_templates': [
                    "Study of {phenomenon} in {system}",
                    "Experimental Investigation of {phenomenon}",
                    "Theoretical Analysis of {system}",
                    "{method} Measurements of {property}"
                ],
                'phenomena': ['quantum entanglement', 'phase transitions', 'wave propagation', 'particle interactions'],
                'systems': ['quantum systems', 'condensed matter', 'optical systems', 'atomic structures'],
                'methods': ['spectroscopic', 'computational', 'experimental', 'theoretical'],
                'properties': ['energy levels', 'magnetic properties', 'optical properties', 'thermal properties'],
                'abstract_template': (
                    "We present a {method} study of {phenomenon} in {system}. "
                    "Using {technique}, we investigate the {property} of the system. "
                    "Our results reveal new insights into the underlying physical mechanisms. "
                    "The findings have implications for understanding {application}. "
                    "These results contribute to the broader field of {domain}."
                )
            },
            
            # Biology
            '2.11': {
                'domain': 'Biology',
                'title_templates': [
                    "Role of {protein} in {process}",
                    "Molecular Mechanisms of {process}",
                    "Genomic Analysis of {organism}",
                    "{technique}-Based Study of {process}"
                ],
                'proteins': ['kinases', 'transcription factors', 'receptors', 'enzymes'],
                'processes': ['cell signaling', 'gene regulation', 'metabolism', 'cell division'],
                'organisms': ['E. coli', 'yeast', 'mammalian cells', 'plant cells'],
                'techniques': ['CRISPR', 'sequencing', 'proteomics', 'microscopy'],
                'abstract_template': (
                    "We investigate the role of {protein} in {process} using {technique}. "
                    "Our study reveals novel molecular mechanisms underlying {process}. "
                    "Results from experiments in {organism} demonstrate that {protein} is essential for {function}. "
                    "These findings provide new insights into cellular {process}. "
                    "This research has potential implications for understanding {application}."
                )
            },
            
            # Medicine
            '2.12': {
                'domain': 'Medicine',
                'title_templates': [
                    "Clinical Trial of {treatment} for {condition}",
                    "Efficacy of {treatment} in {condition} Patients",
                    "Novel Therapeutic Approach for {condition}",
                    "{treatment}-Based Treatment of {condition}"
                ],
                'treatments': ['immunotherapy', 'drug therapy', 'gene therapy', 'combination therapy'],
                'conditions': ['cancer', 'diabetes', 'cardiovascular disease', 'neurological disorders'],
                'methods': ['randomized controlled trial', 'cohort study', 'meta-analysis', 'clinical study'],
                'abstract_template': (
                    "Background: {condition} remains a significant clinical challenge. "
                    "We conducted a {method} to evaluate the efficacy of {treatment} in patients with {condition}. "
                    "Methods: Patients were treated with {treatment} and outcomes were measured. "
                    "Results: The treatment showed significant improvement in {outcome}. "
                    "Conclusions: {treatment} represents a promising therapeutic option for {condition}."
                )
            },
            
            # Chemistry
            '3.13': {
                'domain': 'Chemistry',
                'title_templates': [
                    "Synthesis and Characterization of {compound}",
                    "Catalytic {reaction} using {catalyst}",
                    "Novel {method} for {application}",
                    "Study of {property} in {compound}"
                ],
                'compounds': ['organic molecules', 'metal complexes', 'polymers', 'nanomaterials'],
                'reactions': ['oxidation', 'reduction', 'coupling', 'polymerization'],
                'catalysts': ['metal catalysts', 'organic catalysts', 'enzymatic catalysts'],
                'methods': ['spectroscopy', 'chromatography', 'crystallography'],
                'abstract_template': (
                    "We report the synthesis and characterization of {compound}. "
                    "The {compound} was prepared using {method} and characterized by {technique}. "
                    "Spectroscopic analysis reveals {property}. "
                    "The synthesized compound shows potential for {application} applications. "
                    "This work contributes to the development of new {compound}."
                )
            },
            
            # Mathematics
            '3.11': {
                'domain': 'Mathematics',
                'title_templates': [
                    "On the {problem} of {structure}",
                    "A New Theorem on {topic}",
                    "{method} for Solving {problem}",
                    "Properties of {structure}"
                ],
                'problems': ['convergence', 'stability', 'uniqueness', 'existence'],
                'structures': ['Banach spaces', 'manifolds', 'graphs', 'algebraic structures'],
                'topics': ['differential equations', 'topology', 'number theory', 'optimization'],
                'methods': ['variational', 'iterative', 'analytical', 'numerical'],
                'abstract_template': (
                    "We study the {problem} problem for {structure}. "
                    "Using {method} methods, we prove a new theorem concerning {property}. "
                    "The results extend previous work on {topic} and provide new insights. "
                    "We also discuss applications to {application}. "
                    "This work contributes to the mathematical theory of {topic}."
                )
            },
            
            # Social Sciences
            '2.16': {
                'domain': 'Social Sciences',
                'title_templates': [
                    "Impact of {factor} on {outcome}",
                    "Sociological Analysis of {phenomenon}",
                    "Survey Study of {population}",
                    "{factor} and {outcome}: An Empirical Study"
                ],
                'factors': ['social media', 'economic inequality', 'education', 'urbanization'],
                'outcomes': ['social behavior', 'wellbeing', 'community cohesion', 'development'],
                'phenomena': ['migration patterns', 'cultural change', 'social movements'],
                'populations': ['urban populations', 'youth', 'communities', 'families'],
                'abstract_template': (
                    "This study examines the impact of {factor} on {outcome} in {population}. "
                    "Using {method} methodology, we analyze data from {sample_size} participants. "
                    "Our findings reveal that {factor} significantly affects {outcome}. "
                    "The results have implications for {policy_area} policy. "
                    "This research contributes to understanding {phenomenon} in modern society."
                )
            },
            
            # Psychology
            '2.14': {
                'domain': 'Psychology',
                'title_templates': [
                    "Cognitive Processes in {task}",
                    "Behavioral Study of {phenomenon}",
                    "Neural Correlates of {process}",
                    "{process} and {outcome}: A Psychological Study"
                ],
                'tasks': ['decision making', 'memory recall', 'attention', 'learning'],
                'processes': ['emotion regulation', 'cognitive control', 'perception'],
                'phenomena': ['stress response', 'motivation', 'social cognition'],
                'abstract_template': (
                    "We investigate {phenomenon} using {method} approach. "
                    "Participants completed {task} while {measurement} was recorded. "
                    "Results show that {process} is associated with {outcome}. "
                    "These findings extend current theories of {domain}. "
                    "The study has implications for understanding {application}."
                )
            },
        }
    
    def generate_paper(self, category_code: str) -> Optional[Dict]:
        """
        Generate a synthetic paper for given category
        
        Args:
            category_code: DFG category code
            
        Returns:
            Dictionary with paper metadata
        """
        if category_code not in self.templates:
            # Use a default template
            return self._generate_default_paper(category_code)
        
        template = self.templates[category_code]
        
        # Generate title
        title_template = random.choice(template['title_templates'])
        title_vars = {}
        
        # Extract all possible variables from template
        for key in ['method', 'task', 'technique', 'phenomenon', 'system', 
                    'protein', 'process', 'treatment', 'condition', 'compound',
                    'problem', 'structure', 'factor', 'outcome', 'methods', 'tasks',
                    'techniques', 'phenomena', 'systems', 'properties']:
            if key in template:
                title_vars[key] = random.choice(template[key])
        
        # Format title - fill in any missing variables
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                title = title_template.format(**title_vars)
                break
            except KeyError as e:
                # If a key is missing, use a generic value
                missing_key = str(e).strip("'")
                title_vars[missing_key] = 'research'
        else:
            # Fallback if formatting still fails
            title = f"Research in {template['domain']}"
        
        # Generate abstract
        abstract_template = template['abstract_template']
        abstract_vars = title_vars.copy()
        
        # Add additional variables
        additional_vars = {
            'improvement': random.randint(5, 30),
            'dataset': random.choice(['benchmark', 'standard', 'real-world']),
            'application': random.choice(['practical', 'industrial', 'clinical', 'theoretical']),
            'domain': template['domain'],
            'property': random.choice(template.get('properties', ['characteristics'])),
            'function': random.choice(['regulation', 'control', 'activation', 'inhibition']),
            'outcome': random.choice(['response', 'performance', 'behavior', 'symptoms']),
            'method': random.choice(['experimental', 'computational', 'analytical']),
            'technique': random.choice(['advanced', 'novel', 'state-of-the-art']),
            'measurement': random.choice(['neural activity', 'behavior', 'responses']),
            'sample_size': random.randint(50, 500),
            'policy_area': random.choice(['social', 'economic', 'educational'])
        }
        
        abstract_vars.update(additional_vars)
        
        # Format abstract - fill in any missing variables
        for attempt in range(max_attempts):
            try:
                abstract = abstract_template.format(**abstract_vars)
                break
            except KeyError as e:
                # If a key is missing, use a generic value
                missing_key = str(e).strip("'")
                abstract_vars[missing_key] = 'study'
        else:
            # Fallback if formatting still fails
            abstract = self._generate_simple_abstract(category_code)
        
        # Get category name
        category_name = self.dfg_mapping.get('level_2', {}).get('classes', {}).get(category_code, 'Unknown')
        
        paper = {
            'title': title,
            'abstract': abstract,
            'category': category_code,
            'category_name': category_name,
            'source': 'synthetic',
            'domain': template['domain']
        }
        
        return paper
    
    def _generate_default_paper(self, category_code: str) -> Dict:
        """Generate a default paper when no specific template exists"""
        category_name = self.dfg_mapping.get('level_2', {}).get('classes', {}).get(category_code, 'Unknown Category')
        
        title = f"Research on {category_name}: A Novel Approach"
        abstract = self._generate_simple_abstract(category_code)
        
        return {
            'title': title,
            'abstract': abstract,
            'category': category_code,
            'category_name': category_name,
            'source': 'synthetic',
            'domain': category_name
        }
    
    def _generate_simple_abstract(self, category_code: str) -> str:
        """Generate a simple abstract"""
        category_name = self.dfg_mapping.get('level_2', {}).get('classes', {}).get(category_code, 'the field')
        
        templates = [
            f"This paper presents a novel approach to research in {category_name}. "
            f"We propose a new method that addresses current limitations in the field. "
            f"Our experimental results demonstrate significant improvements over existing approaches. "
            f"The findings contribute to advancing knowledge in {category_name}.",
            
            f"We investigate important questions in {category_name}. "
            f"Through systematic analysis, we develop new insights into key problems. "
            f"Our results have implications for both theory and practice. "
            f"This work opens new directions for future research in {category_name}."
        ]
        
        return random.choice(templates)
    
    def generate_dataset(self, 
                        samples_per_category: int = 100,
                        output_dir: Optional[str] = None) -> List[Dict]:
        """
        Generate complete synthetic dataset
        
        Args:
            samples_per_category: Number of samples to generate per category
            output_dir: Optional output directory to save dataset
            
        Returns:
            List of generated papers
        """
        dataset = []
        
        # Get all Level 2 categories
        level_2_classes = self.dfg_mapping.get('level_2', {}).get('classes', {})
        
        logger.info(f"Generating {samples_per_category} samples for {len(level_2_classes)} categories")
        
        for category_code in level_2_classes.keys():
            for i in range(samples_per_category):
                paper = self.generate_paper(category_code)
                if paper:
                    paper['id'] = f"synthetic_{category_code}_{i:04d}"
                    dataset.append(paper)
        
        logger.info(f"Generated {len(dataset)} synthetic papers")
        
        # Save if output directory specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            output_file = Path(output_dir) / 'synthetic_dataset.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved synthetic dataset to {output_file}")
        
        return dataset


def create_training_ready_dataset(
    synthetic_papers: List[Dict],
    output_dir: str
) -> Dict[str, str]:
    """
    Convert synthetic papers to training-ready format
    
    Args:
        synthetic_papers: List of synthetic papers
        output_dir: Output directory
        
    Returns:
        Labels dictionary mapping paper IDs to categories
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create labels dictionary
    labels = {}
    
    for paper in synthetic_papers:
        paper_id = paper['id']
        category = paper['category']
        labels[paper_id] = category
    
    # Save papers
    papers_file = Path(output_dir) / 'synthetic_papers.json'
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_papers, f, indent=2, ensure_ascii=False)
    
    # Save labels
    labels_file = Path(output_dir) / 'synthetic_labels.json'
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(papers)} papers to {papers_file}")
    logger.info(f"Saved {len(labels)} labels to {labels_file}")
    
    return labels


if __name__ == "__main__":
    # Test synthetic data generation
    logging.basicConfig(level=logging.INFO)
    
    # Load DFG mapping
    import yaml
    with open('data/dfg_mapping.json', 'r', encoding='utf-8') as f:
        dfg_mapping = json.load(f)
    
    # Generate synthetic dataset
    generator = SyntheticPaperGenerator(dfg_mapping)
    
    # Generate a few samples
    print("Generating sample papers...\n")
    for category in ['3.15', '2.11', '3.12']:
        paper = generator.generate_paper(category)
        print(f"Category: {category} - {paper['category_name']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract'][:200]}...")
        print()
    
    # Generate full dataset
    dataset = generator.generate_dataset(samples_per_category=10, output_dir='data/synthetic')
    print(f"\nGenerated {len(dataset)} synthetic papers")

