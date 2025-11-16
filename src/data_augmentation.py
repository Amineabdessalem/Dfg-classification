"""
Data Augmentation Module for DFG Classifier
Includes various augmentation techniques to improve model robustness
"""

import re
import random
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer

# Optional: nlpaug for advanced augmentation (not required)
try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    naw = None
    nas = None

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Text augmentation for scientific texts"""
    
    def __init__(self, 
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 augmentation_prob: float = 0.3):
        """
        Initialize text augmenter
        
        Args:
            model_name: Model name for tokenizer
            augmentation_prob: Probability of applying augmentation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.augmentation_prob = augmentation_prob
        
        # Initialize augmenters (with error handling for missing models)
        self.augmenters = []
        self._initialize_augmenters()
    
    def _initialize_augmenters(self):
        """Initialize various augmenters"""
        if NLPAUG_AVAILABLE:
            try:
                # Synonym replacement using WordNet
                self.augmenters.append(
                    ('synonym', naw.SynonymAug(aug_src='wordnet', aug_p=0.1))
                )
            except Exception as e:
                logger.warning(f"Could not initialize synonym augmenter: {e}")
        else:
            logger.info("nlpaug not available, using built-in augmentation methods only")
        
        # We'll use simpler, more reliable augmenters
        # Contextual word embeddings can be added later if needed
    
    def augment_text(self, text: str, num_aug: int = 1) -> List[str]:
        """
        Augment text using various techniques
        
        Args:
            text: Input text
            num_aug: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_aug):
            # Choose augmentation strategy
            aug_type = random.choice([
                'original',
                'synonym_replacement',
                'random_swap',
                'random_deletion',
                'paraphrase',
                'back_translation_simulation'
            ])
            
            if aug_type == 'original' or random.random() > self.augmentation_prob:
                augmented_texts.append(text)
            elif aug_type == 'synonym_replacement':
                augmented_texts.append(self._synonym_replacement(text))
            elif aug_type == 'random_swap':
                augmented_texts.append(self._random_swap(text))
            elif aug_type == 'random_deletion':
                augmented_texts.append(self._random_deletion(text))
            elif aug_type == 'paraphrase':
                augmented_texts.append(self._simple_paraphrase(text))
            else:
                augmented_texts.append(self._back_translation_simulation(text))
        
        return augmented_texts
    
    def _synonym_replacement(self, text: str, n: int = 3) -> str:
        """Replace n words with synonyms"""
        words = text.split()
        if len(words) == 0:
            return text
        
        # Simple synonym dictionary for scientific terms
        synonyms = {
            'method': ['approach', 'technique', 'procedure'],
            'result': ['outcome', 'finding', 'conclusion'],
            'study': ['research', 'investigation', 'analysis'],
            'data': ['information', 'evidence', 'observations'],
            'model': ['framework', 'system', 'structure'],
            'propose': ['present', 'introduce', 'suggest'],
            'demonstrate': ['show', 'illustrate', 'exhibit'],
            'novel': ['new', 'innovative', 'original'],
            'significant': ['important', 'substantial', 'notable'],
            'performance': ['effectiveness', 'efficiency', 'results']
        }
        
        num_replacements = min(n, len(words))
        indices = random.sample(range(len(words)), num_replacements)
        
        for idx in indices:
            word = words[idx].lower()
            if word in synonyms:
                words[idx] = random.choice(synonyms[word])
        
        return ' '.join(words)
    
    def _random_swap(self, text: str, n: int = 2) -> str:
        """Randomly swap two words in the sentence n times"""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        # If all words are deleted, return original
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple paraphrasing by restructuring sentences"""
        # This is a simplified version - can be enhanced with T5 or other models
        patterns = [
            (r'This paper (presents|proposes) (.*)', r'We \1 \2'),
            (r'We (present|propose) (.*)', r'This study \1s \2'),
            (r'The results show that (.*)', r'Our findings indicate that \1'),
            (r'Our findings indicate that (.*)', r'The results demonstrate that \1'),
        ]
        
        augmented = text
        for pattern, replacement in patterns:
            if re.search(pattern, augmented, re.IGNORECASE):
                augmented = re.sub(pattern, replacement, augmented, flags=re.IGNORECASE)
                break
        
        return augmented
    
    def _back_translation_simulation(self, text: str) -> str:
        """Simulate back translation by paraphrasing"""
        # Simpler version without actual translation
        # In production, use MarianMT or similar
        return self._simple_paraphrase(text)


class DatasetBalancer:
    """Balance dataset across classes"""
    
    def __init__(self, strategy: str = 'oversample'):
        """
        Initialize dataset balancer
        
        Args:
            strategy: Balancing strategy ('oversample', 'undersample', 'smote')
        """
        self.strategy = strategy
    
    def balance_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Balance dataset across classes
        
        Args:
            dataset: List of dataset samples
            
        Returns:
            Balanced dataset
        """
        # Group by class
        class_samples = {}
        for sample in dataset:
            label = sample['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)
        
        # Find max class size
        max_size = max(len(samples) for samples in class_samples.values())
        min_size = min(len(samples) for samples in class_samples.values())
        
        logger.info(f"Class distribution - Min: {min_size}, Max: {max_size}")
        
        balanced_dataset = []
        
        if self.strategy == 'oversample':
            # Oversample minority classes
            for label, samples in class_samples.items():
                if len(samples) < max_size:
                    # Oversample with replacement
                    oversampled = random.choices(samples, k=max_size)
                    balanced_dataset.extend(oversampled)
                else:
                    balanced_dataset.extend(samples)
        
        elif self.strategy == 'undersample':
            # Undersample majority classes
            for label, samples in class_samples.items():
                if len(samples) > min_size:
                    undersampled = random.sample(samples, min_size)
                    balanced_dataset.extend(undersampled)
                else:
                    balanced_dataset.extend(samples)
        
        else:  # 'weighted' - return original with class weights
            balanced_dataset = dataset
        
        random.shuffle(balanced_dataset)
        logger.info(f"Balanced dataset size: {len(balanced_dataset)}")
        
        return balanced_dataset
    
    def get_class_weights(self, dataset: List[Dict]) -> Dict[str, float]:
        """Calculate class weights for weighted loss"""
        from collections import Counter
        
        labels = [sample['label'] for sample in dataset]
        label_counts = Counter(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        class_weights = {
            label: total / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
        
        return class_weights


class MixupAugmenter:
    """Mixup augmentation for text classification"""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup augmenter
        
        Args:
            alpha: Mixup hyperparameter
        """
        self.alpha = alpha
    
    def mixup(self, sample1: Dict, sample2: Dict) -> Dict:
        """
        Apply mixup to two samples
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Mixed sample
        """
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix input embeddings (would be done at model level)
        # Here we just combine texts with a marker
        mixed_text = f"{sample1['combined_text']} [MIX] {sample2['combined_text']}"
        
        # Create mixed sample
        mixed_sample = {
            **sample1,
            'combined_text': mixed_text,
            'mixup_lambda': lam,
            'mixup_label1': sample1['label'],
            'mixup_label2': sample2['label'],
            'is_mixup': True
        }
        
        return mixed_sample


def create_augmented_dataset(
    dataset: List[Dict],
    augmenter: TextAugmenter,
    augmentation_factor: int = 2,
    balance_classes: bool = True
) -> List[Dict]:
    """
    Create augmented dataset
    
    Args:
        dataset: Original dataset
        augmenter: Text augmenter
        augmentation_factor: Number of augmented versions per sample
        balance_classes: Whether to balance classes
        
    Returns:
        Augmented dataset
    """
    augmented_dataset = []
    
    logger.info(f"Creating augmented dataset with factor {augmentation_factor}")
    
    for sample in dataset:
        # Add original sample
        augmented_dataset.append(sample)
        
        # Generate augmented versions
        original_text = sample.get('combined_text', '')
        augmented_texts = augmenter.augment_text(original_text, augmentation_factor - 1)
        
        for aug_text in augmented_texts:
            aug_sample = sample.copy()
            aug_sample['combined_text'] = aug_text
            aug_sample['is_augmented'] = True
            augmented_dataset.append(aug_sample)
    
    # Balance classes if requested
    if balance_classes:
        balancer = DatasetBalancer(strategy='oversample')
        augmented_dataset = balancer.balance_dataset(augmented_dataset)
    
    logger.info(f"Augmented dataset size: {len(augmented_dataset)}")
    
    return augmented_dataset


if __name__ == "__main__":
    # Test augmentation
    logging.basicConfig(level=logging.INFO)
    
    augmenter = TextAugmenter()
    
    test_text = "This paper proposes a novel method for text classification using deep learning."
    augmented = augmenter.augment_text(test_text, num_aug=3)
    
    print("Original:", test_text)
    print("\nAugmented versions:")
    for i, aug in enumerate(augmented, 1):
        print(f"{i}. {aug}")

