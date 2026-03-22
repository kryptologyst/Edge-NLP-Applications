"""Model implementations for Edge NLP Applications."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

logger = logging.getLogger(__name__)


class PyTorchIntentLSTM(nn.Module):
    """PyTorch implementation of LSTM-based intent classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        num_classes: int = 10,
        max_sequence_length: int = 5,
        dropout: float = 0.2
    ):
        """Initialize PyTorch LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of intent classes
            max_sequence_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        
        # Model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Fully connected layers
        x = self.dropout(last_hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self) -> int:
        """Get total number of parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


class TensorFlowIntentLSTM:
    """TensorFlow implementation of LSTM-based intent classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        num_classes: int = 10,
        max_sequence_length: int = 5,
        dropout: float = 0.2
    ):
        """Initialize TensorFlow LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of intent classes
            max_sequence_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the TensorFlow model.
        
        Returns:
            Compiled TensorFlow model
        """
        model = models.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            layers.LSTM(self.hidden_dim, dropout=self.dropout),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            X_train: Training input data
            y_train: Training labels
            X_val: Validation input data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        callbacks_list = []
        
        # Early stopping
        if X_val is not None and y_val is not None:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks_list.append(early_stopping)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Evaluate model performance.
        
        Args:
            X: Test input data
            y: Test labels
            
        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(X, y, verbose=0)
    
    def save_model(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        self.model.save(path)
    
    def load_model(self, path: str) -> None:
        """Load model from file.
        
        Args:
            path: Path to load model from
        """
        self.model = tf.keras.models.load_model(path)
    
    def get_model_size(self) -> int:
        """Get total number of parameters.
        
        Returns:
            Total number of parameters
        """
        return self.model.count_params()


class QuantizedIntentLSTM:
    """Quantized version of LSTM intent classifier."""
    
    def __init__(self, base_model: Union[PyTorchIntentLSTM, TensorFlowIntentLSTM]):
        """Initialize quantized model.
        
        Args:
            base_model: Base model to quantize
        """
        self.base_model = base_model
        self.quantized_model = None
    
    def quantize_pytorch(self, calibration_data: torch.Tensor) -> None:
        """Quantize PyTorch model using dynamic quantization.
        
        Args:
            calibration_data: Data for calibration
        """
        if not isinstance(self.base_model, PyTorchIntentLSTM):
            raise ValueError("Base model must be PyTorchIntentLSTM for PyTorch quantization")
        
        # Set model to evaluation mode
        self.base_model.eval()
        
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.base_model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )
        
        logger.info("PyTorch model quantized successfully")
    
    def quantize_tensorflow(self, representative_dataset: np.ndarray) -> None:
        """Quantize TensorFlow model using TensorFlow Lite.
        
        Args:
            representative_dataset: Representative dataset for calibration
        """
        if not isinstance(self.base_model, TensorFlowIntentLSTM):
            raise ValueError("Base model must be TensorFlowIntentLSTM for TensorFlow quantization")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.base_model.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for calibration
        def representative_data_gen():
            for i in range(len(representative_dataset)):
                yield [representative_dataset[i:i+1]]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert model
        self.quantized_model = converter.convert()
        
        logger.info("TensorFlow model quantized successfully")
    
    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Make predictions with quantized model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.quantized_model is None:
            raise ValueError("Model must be quantized before making predictions")
        
        if isinstance(self.base_model, PyTorchIntentLSTM):
            with torch.no_grad():
                return self.quantized_model(X)
        else:
            # For TensorFlow Lite
            interpreter = tf.lite.Interpreter(model_content=self.quantized_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], X.astype(np.int8))
            interpreter.invoke()
            
            return interpreter.get_tensor(output_details[0]['index'])


def create_model(
    framework: str,
    vocab_size: int,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    num_classes: int = 10,
    max_sequence_length: int = 5,
    dropout: float = 0.2
) -> Union[PyTorchIntentLSTM, TensorFlowIntentLSTM]:
    """Create a model instance.
    
    Args:
        framework: Framework to use ("pytorch" or "tensorflow")
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_classes: Number of intent classes
        max_sequence_length: Maximum sequence length
        dropout: Dropout rate
        
    Returns:
        Model instance
    """
    if framework.lower() == "pytorch":
        return PyTorchIntentLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
    elif framework.lower() == "tensorflow":
        return TensorFlowIntentLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")
