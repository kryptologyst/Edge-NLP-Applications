"""Evaluation and metrics for Edge NLP Applications."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import torch
import tensorflow as tf

logger = logging.getLogger(__name__)


class EdgeNLPEvaluator:
    """Comprehensive evaluator for Edge NLP models."""
    
    def __init__(self, class_names: List[str]):
        """Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        framework: str = "tensorflow"
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test input data
            y_test: Test labels
            framework: Framework used ("pytorch" or "tensorflow")
            
        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation...")
        
        # Get predictions
        start_time = time.time()
        if framework.lower() == "pytorch":
            predictions = self._predict_pytorch(model, X_test)
        else:
            predictions = self._predict_tensorflow(model, X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        # Per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            y_test, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(X_test),
            'confusion_matrix': cm.tolist(),
            'per_class_precision': per_class_metrics[0].tolist(),
            'per_class_recall': per_class_metrics[1].tolist(),
            'per_class_f1': per_class_metrics[2].tolist(),
            'support': per_class_metrics[3].tolist(),
            'class_names': self.class_names
        }
        
        self.results = results
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        
        return results
    
    def _predict_pytorch(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from PyTorch model.
        
        Args:
            model: PyTorch model
            X_test: Test input data
            
        Returns:
            Predictions
        """
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            outputs = model(X_test_tensor)
            predictions = outputs.argmax(dim=1).numpy()
        
        return predictions
    
    def _predict_tensorflow(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from TensorFlow model.
        
        Args:
            model: TensorFlow model
            X_test: Test input data
            
        Returns:
            Predictions
        """
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
            return np.argmax(predictions, axis=1)
        else:
            # For TensorFlow Lite models
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            predictions = []
            for i in range(len(X_test)):
                interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1])
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(np.argmax(output))
            
            return np.array(predictions)
    
    def plot_confusion_matrix(
        self, 
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            save_path: Path to save plot (optional)
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_class_metrics(
        self, 
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot per-class metrics.
        
        Args:
            save_path: Path to save plot (optional)
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1']
        titles = ['Precision', 'Recall', 'F1-Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            values = self.results[metric]
            axes[i].bar(self.class_names, values)
            axes[i].set_title(title)
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class metrics plot saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self) -> str:
        """Generate detailed classification report.
        
        Returns:
            Classification report string
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        # Create a mock y_true and y_pred for sklearn's classification_report
        # This is a workaround since we already have the metrics
        y_true = np.array([0] * self.results['support'][0])  # Dummy data
        y_pred = np.array([0] * self.results['support'][0])  # Dummy data
        
        # Build report manually
        report = f"Classification Report\n"
        report += f"{'='*50}\n"
        report += f"Overall Accuracy: {self.results['accuracy']:.4f}\n"
        report += f"Weighted Precision: {self.results['precision']:.4f}\n"
        report += f"Weighted Recall: {self.results['recall']:.4f}\n"
        report += f"Weighted F1-Score: {self.results['f1_score']:.4f}\n\n"
        
        report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
        report += f"{'-'*60}\n"
        
        for i, class_name in enumerate(self.class_names):
            precision = self.results['per_class_precision'][i]
            recall = self.results['per_class_recall'][i]
            f1 = self.results['per_class_f1'][i]
            support = self.results['support'][i]
            
            report += f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}\n"
        
        return report
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")


class PerformanceBenchmark:
    """Benchmark performance metrics for edge deployment."""
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.benchmark_results = {}
    
    def benchmark_inference_speed(
        self,
        model: Any,
        X_test: np.ndarray,
        framework: str = "tensorflow",
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            model: Trained model
            X_test: Test data
            framework: Framework used
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking inference speed with {num_runs} runs...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            if framework.lower() == "pytorch":
                self._predict_pytorch(model, X_test[:1])
            else:
                self._predict_tensorflow(model, X_test[:1])
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            if framework.lower() == "pytorch":
                self._predict_pytorch(model, X_test)
            else:
                self._predict_tensorflow(model, X_test)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times_ms = np.array(times) * 1000
        per_sample_times_ms = times_ms / len(X_test)
        
        results = {
            'total_time_mean_ms': np.mean(times_ms),
            'total_time_std_ms': np.std(times_ms),
            'total_time_p50_ms': np.percentile(times_ms, 50),
            'total_time_p95_ms': np.percentile(times_ms, 95),
            'total_time_p99_ms': np.percentile(times_ms, 99),
            'per_sample_time_mean_ms': np.mean(per_sample_times_ms),
            'per_sample_time_std_ms': np.std(per_sample_times_ms),
            'per_sample_time_p50_ms': np.percentile(per_sample_times_ms, 50),
            'per_sample_time_p95_ms': np.percentile(per_sample_times_ms, 95),
            'per_sample_time_p99_ms': np.percentile(per_sample_times_ms, 99),
            'throughput_samples_per_sec': len(X_test) / np.mean(times),
            'num_runs': num_runs
        }
        
        self.benchmark_results['inference_speed'] = results
        logger.info(f"Benchmark completed. Mean per-sample time: {results['per_sample_time_mean_ms']:.2f}ms")
        
        return results
    
    def _predict_pytorch(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from PyTorch model."""
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            outputs = model(X_test_tensor)
            return outputs.argmax(dim=1).numpy()
    
    def _predict_tensorflow(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from TensorFlow model."""
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test, verbose=0)
            return np.argmax(predictions, axis=1)
        else:
            # For TensorFlow Lite models
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            predictions = []
            for i in range(len(X_test)):
                interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1])
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(np.argmax(output))
            
            return np.array(predictions)
    
    def create_performance_report(self) -> str:
        """Create performance benchmark report.
        
        Returns:
            Performance report string
        """
        if not self.benchmark_results:
            return "No benchmark results available."
        
        report = "Performance Benchmark Report\n"
        report += "=" * 50 + "\n\n"
        
        for benchmark_type, results in self.benchmark_results.items():
            report += f"{benchmark_type.replace('_', ' ').title()}:\n"
            report += "-" * 30 + "\n"
            
            for metric, value in results.items():
                if isinstance(value, float):
                    report += f"{metric}: {value:.4f}\n"
                else:
                    report += f"{metric}: {value}\n"
            
            report += "\n"
        
        return report
    
    def save_benchmark_results(self, output_path: Union[str, Path]) -> None:
        """Save benchmark results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")


def create_leaderboard(
    results: List[Dict[str, Any]], 
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Create a leaderboard from multiple model results.
    
    Args:
        results: List of model evaluation results
        output_path: Path to save leaderboard (optional)
        
    Returns:
        Leaderboard DataFrame
    """
    leaderboard_data = []
    
    for i, result in enumerate(results):
        leaderboard_data.append({
            'Model': f"Model_{i+1}",
            'Accuracy': result.get('accuracy', 0.0),
            'Precision': result.get('precision', 0.0),
            'Recall': result.get('recall', 0.0),
            'F1-Score': result.get('f1_score', 0.0),
            'Inference_Time_ms': result.get('inference_time_per_sample', 0.0),
            'Model_Size_MB': result.get('model_size_mb', 0.0),
            'Framework': result.get('framework', 'unknown')
        })
    
    leaderboard = pd.DataFrame(leaderboard_data)
    leaderboard = leaderboard.sort_values('Accuracy', ascending=False)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_path, index=False)
        logger.info(f"Leaderboard saved to {output_path}")
    
    return leaderboard
