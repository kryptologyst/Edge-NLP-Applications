"""Data processing utilities for Edge NLP Applications."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

logger = logging.getLogger(__name__)


class IntentDataset:
    """Dataset class for intent recognition tasks."""
    
    def __init__(
        self,
        sentences: List[str],
        labels: List[str],
        max_sequence_length: int = 5,
        vocab_size: int = 1000,
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        """Initialize intent dataset.
        
        Args:
            sentences: List of input sentences
            labels: List of corresponding intent labels
            max_sequence_length: Maximum sequence length for padding
            vocab_size: Vocabulary size for tokenizer
            test_size: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
        """
        self.sentences = sentences
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Initialize tokenizer and label encoder
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        
        # Process data
        self._process_data()
    
    def _process_data(self) -> None:
        """Process the raw data into trainable format."""
        logger.info(f"Processing {len(self.sentences)} samples")
        
        # Fit tokenizer on all sentences
        self.tokenizer.fit_on_texts(self.sentences)
        
        # Convert sentences to sequences
        sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        # Encode labels
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Split data
        self._split_data()
        
        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Training samples: {len(self.X_train)}")
        logger.info(f"Test samples: {len(self.X_test)}")
    
    def _split_data(self) -> None:
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.padded_sequences,
            self.encoded_labels,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=self.encoded_labels
        )
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data.
        
        Returns:
            Tuple of (X_train, y_train)
        """
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        return self.X_test, self.y_test
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return len(self.tokenizer.word_index) + 1
    
    def get_num_classes(self) -> int:
        """Get number of classes.
        
        Returns:
            Number of classes
        """
        return len(self.label_encoder.classes_)
    
    def get_class_names(self) -> List[str]:
        """Get class names.
        
        Returns:
            List of class names
        """
        return list(self.label_encoder.classes_)
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """Preprocess a single text input.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed sequence
        """
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            sequence, 
            maxlen=self.max_sequence_length, 
            padding='post', 
            truncating='post'
        )
        return padded[0]
    
    def decode_prediction(self, prediction: int) -> str:
        """Decode prediction to class name.
        
        Args:
            prediction: Encoded prediction
            
        Returns:
            Class name
        """
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def save_tokenizer(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'word_index': self.tokenizer.word_index,
            'num_words': self.tokenizer.num_words,
            'oov_token': self.tokenizer.oov_token
        }
        
        with open(path, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
    
    def save_label_encoder(self, path: Union[str, Path]) -> None:
        """Save label encoder to file.
        
        Args:
            path: Path to save label encoder
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        encoder_data = {
            'classes': self.label_encoder.classes_.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(encoder_data, f, indent=2)


def create_synthetic_dataset(
    num_samples: int = 1000,
    num_intents: int = 10,
    max_sequence_length: int = 5,
    random_seed: int = 42
) -> IntentDataset:
    """Create a synthetic dataset for testing purposes.
    
    Args:
        num_samples: Number of samples to generate
        num_intents: Number of different intents
        max_sequence_length: Maximum sequence length
        random_seed: Random seed for reproducibility
        
    Returns:
        IntentDataset object
    """
    np.random.seed(random_seed)
    
    # Define intent templates
    intent_templates = {
        "lights_on": ["turn on lights", "switch on lights", "lights on", "illuminate room"],
        "lights_off": ["turn off lights", "switch off lights", "lights off", "darken room"],
        "fan_on": ["turn on fan", "start fan", "fan on", "activate fan"],
        "fan_off": ["turn off fan", "stop fan", "fan off", "deactivate fan"],
        "music_play": ["play music", "start music", "music on", "begin song"],
        "music_stop": ["stop music", "pause music", "music off", "end song"],
        "volume_up": ["increase volume", "volume up", "louder", "turn up sound"],
        "volume_down": ["decrease volume", "volume down", "quieter", "turn down sound"],
        "weather_query": ["what's weather", "weather report", "check weather", "weather today"],
        "set_alarm": ["set alarm", "create alarm", "schedule alarm", "wake up time"]
    }
    
    # Generate samples
    sentences = []
    labels = []
    
    intent_names = list(intent_templates.keys())[:num_intents]
    
    for _ in range(num_samples):
        intent = np.random.choice(intent_names)
        template = np.random.choice(intent_templates[intent])
        
        # Add some variation to the template
        variations = [
            template,
            f"please {template}",
            f"can you {template}",
            f"i want to {template}",
            f"help me {template}"
        ]
        
        sentence = np.random.choice(variations)
        sentences.append(sentence)
        labels.append(intent)
    
    return IntentDataset(
        sentences=sentences,
        labels=labels,
        max_sequence_length=max_sequence_length,
        test_size=0.2,
        random_seed=random_seed
    )


def load_dataset_from_file(file_path: Union[str, Path]) -> IntentDataset:
    """Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON file containing sentences and labels
        
    Returns:
        IntentDataset object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return IntentDataset(
        sentences=data['sentences'],
        labels=data['labels'],
        max_sequence_length=data.get('max_sequence_length', 5),
        test_size=data.get('test_size', 0.2),
        random_seed=data.get('random_seed', 42)
    )


def save_dataset_to_file(
    dataset: IntentDataset, 
    file_path: Union[str, Path]
) -> None:
    """Save dataset to JSON file.
    
    Args:
        dataset: IntentDataset object to save
        file_path: Path to save the dataset
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'sentences': dataset.sentences,
        'labels': dataset.labels,
        'max_sequence_length': dataset.max_sequence_length,
        'test_size': dataset.test_size,
        'random_seed': dataset.random_seed
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
