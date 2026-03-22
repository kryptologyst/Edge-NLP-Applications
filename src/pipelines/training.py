"""Training pipeline for Edge NLP Applications."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from omegaconf import DictConfig

from ..utils.core import PerformanceMonitor, get_device, set_deterministic_seed
from ..utils.data_utils import IntentDataset
from .intent_classifier import PyTorchIntentLSTM, TensorFlowIntentLSTM, create_model

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline for intent classification models."""
    
    def __init__(self, config: DictConfig):
        """Initialize training pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device(config.deployment.target_device)
        self.monitor = PerformanceMonitor(
            enable_memory=config.monitoring.log_memory,
            enable_latency=config.monitoring.log_latency
        )
        
        # Set random seeds for reproducibility
        set_deterministic_seed(config.data.random_seed)
        
        # Initialize model and dataset
        self.model = None
        self.dataset = None
        self.training_history = None
        
    def prepare_data(self, dataset: IntentDataset) -> None:
        """Prepare dataset for training.
        
        Args:
            dataset: IntentDataset object
        """
        self.dataset = dataset
        logger.info(f"Dataset prepared with {len(dataset.sentences)} samples")
    
    def create_model(self, framework: str = "tensorflow") -> None:
        """Create model instance.
        
        Args:
            framework: Framework to use ("pytorch" or "tensorflow")
        """
        self.model = create_model(
            framework=framework,
            vocab_size=self.dataset.get_vocab_size(),
            embedding_dim=self.config.model.embedding_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_classes=self.dataset.get_num_classes(),
            max_sequence_length=self.config.model.max_sequence_length,
            dropout=self.config.model.dropout
        )
        
        logger.info(f"Model created using {framework} framework")
        logger.info(f"Model parameters: {self.model.get_model_size()}")
    
    def train_pytorch(self) -> Dict[str, List[float]]:
        """Train PyTorch model.
        
        Returns:
            Training history
        """
        if not isinstance(self.model, PyTorchIntentLSTM):
            raise ValueError("Model must be PyTorchIntentLSTM for PyTorch training")
        
        # Get training data
        X_train, y_train = self.dataset.get_training_data()
        X_test, y_test = self.dataset.get_test_data()
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=True
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.training.learning_rate
        )
        
        # Training loop
        self.model.train()
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.training.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_total += target.size(0)
            
            # Calculate metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = epoch_correct / epoch_total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            # Validation
            self.model.eval()
            val_loss, val_acc = self._evaluate_pytorch(X_test_tensor, y_test_tensor, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            self.model.train()
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, "
                    f"Train Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
        
        self.training_history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        return self.training_history
    
    def train_tensorflow(self) -> tf.keras.callbacks.History:
        """Train TensorFlow model.
        
        Returns:
            Training history
        """
        if not isinstance(self.model, TensorFlowIntentLSTM):
            raise ValueError("Model must be TensorFlowIntentLSTM for TensorFlow training")
        
        # Get training data
        X_train, y_train = self.dataset.get_training_data()
        X_test, y_test = self.dataset.get_test_data()
        
        # Train model
        history = self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            verbose=1
        )
        
        self.training_history = history.history
        return history
    
    def _evaluate_pytorch(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate PyTorch model.
        
        Args:
            X: Input data
            y: Target labels
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        with torch.no_grad():
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = criterion(output, y)
            pred = output.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            accuracy = correct / y.size(0)
        
        return loss.item(), accuracy
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance.
        
        Returns:
            Evaluation metrics
        """
        X_test, y_test = self.dataset.get_test_data()
        
        if isinstance(self.model, PyTorchIntentLSTM):
            return self._evaluate_pytorch_detailed(X_test, y_test)
        else:
            return self._evaluate_tensorflow_detailed(X_test, y_test)
    
    def _evaluate_pytorch_detailed(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Detailed evaluation for PyTorch model.
        
        Args:
            X_test: Test input data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = outputs.argmax(dim=1)
            
            # Calculate metrics
            correct = predictions.eq(y_test_tensor).sum().item()
            accuracy = correct / len(y_test)
            
            # Calculate per-class metrics
            class_correct = torch.zeros(self.dataset.get_num_classes())
            class_total = torch.zeros(self.dataset.get_num_classes())
            
            for i in range(len(y_test)):
                label = y_test_tensor[i]
                class_correct[label] += predictions[i] == label
                class_total[label] += 1
            
            class_accuracies = class_correct / class_total
            avg_class_accuracy = class_accuracies.mean().item()
        
        return {
            'accuracy': accuracy,
            'avg_class_accuracy': avg_class_accuracy,
            'num_test_samples': len(y_test)
        }
    
    def _evaluate_tensorflow_detailed(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Detailed evaluation for TensorFlow model.
        
        Args:
            X_test: Test input data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Get predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == y_test)
        
        # Calculate per-class metrics
        class_accuracies = []
        for i in range(self.dataset.get_num_classes()):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == y_test[class_mask])
                class_accuracies.append(class_acc)
        
        avg_class_accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
        
        return {
            'accuracy': accuracy,
            'avg_class_accuracy': avg_class_accuracy,
            'num_test_samples': len(y_test)
        }
    
    def save_model(self, output_path: Union[str, Path]) -> None:
        """Save trained model.
        
        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, PyTorchIntentLSTM):
            torch.save(self.model.state_dict(), output_path)
        else:
            self.model.save_model(str(output_path))
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load trained model.
        
        Args:
            model_path: Path to load model from
        """
        model_path = Path(model_path)
        
        if isinstance(self.model, PyTorchIntentLSTM):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            self.model.load_model(str(model_path))
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            'framework': 'pytorch' if isinstance(self.model, PyTorchIntentLSTM) else 'tensorflow',
            'vocab_size': self.dataset.get_vocab_size(),
            'num_classes': self.dataset.get_num_classes(),
            'model_size': self.model.get_model_size(),
            'model_size_mb': self.model.get_model_size() * 4 / (1024 * 1024),  # Assuming float32
            'device': self.device,
            'config': self.config
        }
