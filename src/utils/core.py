"""Core utilities for Edge NLP Applications."""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # TensorFlow settings for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_device(device_type: str = "auto") -> str:
    """Get the appropriate device for computation.
    
    Args:
        device_type: Device type ("auto", "cpu", "gpu", "cuda")
        
    Returns:
        Device string for PyTorch/TensorFlow
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_type


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_model_size_mb(model: Union[torch.nn.Module, tf.keras.Model]) -> float:
    """Calculate model size in megabytes.
    
    Args:
        model: PyTorch or TensorFlow model
        
    Returns:
        Model size in MB
    """
    if isinstance(model, torch.nn.Module):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
    elif isinstance(model, tf.keras.Model):
        total_size = model.count_params() * 4  # Assuming float32
    else:
        raise ValueError("Unsupported model type")
    
    return total_size / (1024 * 1024)  # Convert to MB


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def validate_input_shape(input_shape: Tuple[int, ...], expected_shape: Tuple[int, ...]) -> bool:
    """Validate input shape against expected shape.
    
    Args:
        input_shape: Actual input shape
        expected_shape: Expected input shape
        
    Returns:
        True if shapes are compatible
    """
    if len(input_shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(input_shape, expected_shape):
        if expected != -1 and actual != expected:
            return False
    
    return True


class PerformanceMonitor:
    """Monitor performance metrics during inference."""
    
    def __init__(self, enable_memory: bool = True, enable_latency: bool = True):
        """Initialize performance monitor.
        
        Args:
            enable_memory: Enable memory monitoring
            enable_latency: Enable latency monitoring
        """
        self.enable_memory = enable_memory
        self.enable_latency = enable_latency
        self.latencies: List[float] = []
        self.memory_usage: List[float] = []
        
    def start_timer(self) -> float:
        """Start timing measurement.
        
        Returns:
            Start time
        """
        return torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()
    
    def end_timer(self, start_time: Union[torch.cuda.Event, float]) -> float:
        """End timing measurement.
        
        Args:
            start_time: Start time from start_timer()
            
        Returns:
            Elapsed time in milliseconds
        """
        if isinstance(start_time, torch.cuda.Event):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            return start_time.elapsed_time(end_time)
        else:
            import time
            return (time.time() - start_time) * 1000
    
    def record_metrics(self, latency_ms: float, memory_mb: Optional[float] = None) -> None:
        """Record performance metrics.
        
        Args:
            latency_ms: Inference latency in milliseconds
            memory_mb: Memory usage in MB (optional)
        """
        self.latencies.append(latency_ms)
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {}
        
        if self.latencies:
            stats.update({
                "latency_mean_ms": np.mean(self.latencies),
                "latency_std_ms": np.std(self.latencies),
                "latency_p50_ms": np.percentile(self.latencies, 50),
                "latency_p95_ms": np.percentile(self.latencies, 95),
                "latency_p99_ms": np.percentile(self.latencies, 99),
            })
        
        if self.memory_usage:
            stats.update({
                "memory_mean_mb": np.mean(self.memory_usage),
                "memory_max_mb": np.max(self.memory_usage),
            })
        
        return stats
