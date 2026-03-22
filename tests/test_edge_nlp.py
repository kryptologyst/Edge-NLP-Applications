"""Tests for Edge NLP Applications."""

import pytest
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.core import set_deterministic_seed, get_device, get_model_size_mb
from src.utils.data_utils import IntentDataset, create_synthetic_dataset
from src.models.intent_classifier import PyTorchIntentLSTM, TensorFlowIntentLSTM, create_model
from src.utils.evaluation import EdgeNLPEvaluator, PerformanceBenchmark


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seed setting."""
        set_deterministic_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        np.random.seed(42)
        val2 = np.random.random()
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device == "cpu"
        
        device = get_device("auto")
        assert device in ["cpu", "cuda", "mps"]
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        # Test PyTorch model
        model = torch.nn.Linear(10, 5)
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        
        # Test TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(10,))
        ])
        size_mb = get_model_size_mb(model)
        assert size_mb > 0


class TestDataUtils:
    """Test data utility functions."""
    
    def test_intent_dataset(self):
        """Test IntentDataset class."""
        sentences = ["turn on lights", "play music", "stop music"]
        labels = ["lights_on", "music_play", "music_stop"]
        
        dataset = IntentDataset(sentences, labels)
        
        assert len(dataset.sentences) == 3
        assert len(dataset.labels) == 3
        assert dataset.get_vocab_size() > 0
        assert dataset.get_num_classes() == 3
        
        # Test preprocessing
        processed = dataset.preprocess_text("turn on lights")
        assert isinstance(processed, np.ndarray)
        assert len(processed) == dataset.max_sequence_length
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_samples=100, num_intents=5)
        
        assert len(dataset.sentences) == 100
        assert len(dataset.labels) == 100
        assert dataset.get_num_classes() == 5
        
        # Check that all samples have valid labels
        unique_labels = set(dataset.labels)
        assert len(unique_labels) <= 5


class TestModels:
    """Test model implementations."""
    
    def test_pytorch_intent_lstm(self):
        """Test PyTorch LSTM model."""
        model = PyTorchIntentLSTM(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=32,
            num_classes=5,
            max_sequence_length=5
        )
        
        # Test forward pass
        input_tensor = torch.randint(0, 100, (2, 5))
        output = model(input_tensor)
        
        assert output.shape == (2, 5)
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-6)
        
        # Test model size
        size = model.get_model_size()
        assert size > 0
    
    def test_tensorflow_intent_lstm(self):
        """Test TensorFlow LSTM model."""
        model = TensorFlowIntentLSTM(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=32,
            num_classes=5,
            max_sequence_length=5
        )
        
        # Test model creation
        assert model.model is not None
        
        # Test model size
        size = model.get_model_size()
        assert size > 0
    
    def test_create_model(self):
        """Test model creation function."""
        # Test PyTorch model creation
        pytorch_model = create_model("pytorch", vocab_size=100, num_classes=5)
        assert isinstance(pytorch_model, PyTorchIntentLSTM)
        
        # Test TensorFlow model creation
        tf_model = create_model("tensorflow", vocab_size=100, num_classes=5)
        assert isinstance(tf_model, TensorFlowIntentLSTM)
        
        # Test invalid framework
        with pytest.raises(ValueError):
            create_model("invalid", vocab_size=100, num_classes=5)


class TestEvaluation:
    """Test evaluation functions."""
    
    def test_edge_nlp_evaluator(self):
        """Test EdgeNLPEvaluator class."""
        evaluator = EdgeNLPEvaluator(["class1", "class2", "class3"])
        
        # Create dummy data
        X_test = np.random.randint(0, 100, (10, 5))
        y_test = np.random.randint(0, 3, 10)
        
        # Create dummy model
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X), 3)
        
        model = DummyModel()
        
        # Test evaluation
        results = evaluator.evaluate_model(model, X_test, y_test, "tensorflow")
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "confusion_matrix" in results
    
    def test_performance_benchmark(self):
        """Test PerformanceBenchmark class."""
        benchmark = PerformanceBenchmark()
        
        # Create dummy model
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X), 3)
        
        model = DummyModel()
        X_test = np.random.randint(0, 100, (10, 5))
        
        # Test benchmarking
        results = benchmark.benchmark_inference_speed(model, X_test, "tensorflow", 10, 2)
        
        assert "mean_latency_ms" in results
        assert "throughput_samples_per_sec" in results
        assert results["num_runs"] == 10


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pytorch(self):
        """Test end-to-end PyTorch pipeline."""
        # Create dataset
        dataset = create_synthetic_dataset(num_samples=50, num_intents=3)
        
        # Create model
        model = PyTorchIntentLSTM(
            vocab_size=dataset.get_vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_classes=dataset.get_num_classes(),
            max_sequence_length=5
        )
        
        # Get test data
        X_test, y_test = dataset.get_test_data()
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            output = model(X_test_tensor)
            predictions = output.argmax(dim=1).numpy()
        
        assert len(predictions) == len(y_test)
        assert all(0 <= pred < dataset.get_num_classes() for pred in predictions)
    
    def test_end_to_end_tensorflow(self):
        """Test end-to-end TensorFlow pipeline."""
        # Create dataset
        dataset = create_synthetic_dataset(num_samples=50, num_intents=3)
        
        # Create model
        model = TensorFlowIntentLSTM(
            vocab_size=dataset.get_vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_classes=dataset.get_num_classes(),
            max_sequence_length=5
        )
        
        # Get test data
        X_test, y_test = dataset.get_test_data()
        
        # Test prediction
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        assert len(predicted_classes) == len(y_test)
        assert all(0 <= pred < dataset.get_num_classes() for pred in predicted_classes)


if __name__ == "__main__":
    pytest.main([__file__])
