"""Export and deployment pipeline for Edge NLP Applications."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import tensorflow as tf
import onnx
import onnxruntime as ort
from omegaconf import DictConfig

from ..utils.core import get_device, PerformanceMonitor
from .intent_classifier import PyTorchIntentLSTM, TensorFlowIntentLSTM

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export models to various edge deployment formats."""
    
    def __init__(self, config: DictConfig):
        """Initialize model exporter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.monitor = PerformanceMonitor()
        self.exported_models = {}
    
    def export_pytorch_to_onnx(
        self,
        model: PyTorchIntentLSTM,
        sample_input: torch.Tensor,
        output_path: Union[str, Path],
        opset_version: int = 11
    ) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            output_path: Output path for ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting PyTorch model to ONNX...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        self.exported_models['onnx'] = str(output_path)
        logger.info(f"PyTorch model exported to ONNX: {output_path}")
        
        return str(output_path)
    
    def export_tensorflow_to_tflite(
        self,
        model: TensorFlowIntentLSTM,
        output_path: Union[str, Path],
        quantization: bool = True,
        representative_dataset: Optional[np.ndarray] = None
    ) -> str:
        """Export TensorFlow model to TensorFlow Lite format.
        
        Args:
            model: TensorFlow model
            output_path: Output path for TFLite model
            quantization: Enable quantization
            representative_dataset: Dataset for quantization calibration
            
        Returns:
            Path to exported TFLite model
        """
        logger.info("Exporting TensorFlow model to TensorFlow Lite...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
        
        if quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                def representative_data_gen():
                    for i in range(len(representative_dataset)):
                        yield [representative_dataset[i:i+1]]
                
                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        self.exported_models['tflite'] = str(output_path)
        logger.info(f"TensorFlow model exported to TFLite: {output_path}")
        
        return str(output_path)
    
    def export_to_coreml(
        self,
        model: TensorFlowIntentLSTM,
        output_path: Union[str, Path]
    ) -> str:
        """Export TensorFlow model to CoreML format.
        
        Args:
            model: TensorFlow model
            output_path: Output path for CoreML model
            
        Returns:
            Path to exported CoreML model
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("coremltools is required for CoreML export")
        
        logger.info("Exporting TensorFlow model to CoreML...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            model.model,
            inputs=[ct.TensorType(shape=(1, self.config.model.max_sequence_length))]
        )
        
        # Save model
        coreml_model.save(str(output_path))
        
        self.exported_models['coreml'] = str(output_path)
        logger.info(f"TensorFlow model exported to CoreML: {output_path}")
        
        return str(output_path)
    
    def benchmark_exported_models(
        self,
        test_data: np.ndarray,
        framework: str = "tensorflow"
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark exported models.
        
        Args:
            test_data: Test data for benchmarking
            framework: Original framework used
            
        Returns:
            Benchmark results for each exported model
        """
        logger.info("Benchmarking exported models...")
        
        benchmark_results = {}
        
        for format_name, model_path in self.exported_models.items():
            logger.info(f"Benchmarking {format_name} model...")
            
            if format_name == 'onnx':
                results = self._benchmark_onnx_model(model_path, test_data)
            elif format_name == 'tflite':
                results = self._benchmark_tflite_model(model_path, test_data)
            elif format_name == 'coreml':
                results = self._benchmark_coreml_model(model_path, test_data)
            else:
                continue
            
            benchmark_results[format_name] = results
        
        return benchmark_results
    
    def _benchmark_onnx_model(
        self, 
        model_path: str, 
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input': test_data[:1]})
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            session.run(None, {'input': test_data})
            end_time = time.time()
            times.append(end_time - start_time)
        
        times_ms = np.array(times) * 1000
        
        return {
            'mean_latency_ms': np.mean(times_ms),
            'std_latency_ms': np.std(times_ms),
            'p50_latency_ms': np.percentile(times_ms, 50),
            'p95_latency_ms': np.percentile(times_ms, 95),
            'p99_latency_ms': np.percentile(times_ms, 99),
            'throughput_samples_per_sec': len(test_data) / np.mean(times)
        }
    
    def _benchmark_tflite_model(
        self, 
        model_path: str, 
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """Benchmark TensorFlow Lite model."""
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_data[:1])
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_data)
            interpreter.invoke()
            end_time = time.time()
            times.append(end_time - start_time)
        
        times_ms = np.array(times) * 1000
        
        return {
            'mean_latency_ms': np.mean(times_ms),
            'std_latency_ms': np.std(times_ms),
            'p50_latency_ms': np.percentile(times_ms, 50),
            'p95_latency_ms': np.percentile(times_ms, 95),
            'p99_latency_ms': np.percentile(times_ms, 99),
            'throughput_samples_per_sec': len(test_data) / np.mean(times)
        }
    
    def _benchmark_coreml_model(
        self, 
        model_path: str, 
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """Benchmark CoreML model."""
        try:
            import coremltools as ct
        except ImportError:
            return {'error': 'coremltools not available'}
        
        # Load CoreML model
        model = ct.models.MLModel(model_path)
        
        # Prepare input
        input_data = {model.input_description[0].name: test_data}
        
        # Warmup
        for _ in range(10):
            model.predict(input_data)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            model.predict(input_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times_ms = np.array(times) * 1000
        
        return {
            'mean_latency_ms': np.mean(times_ms),
            'std_latency_ms': np.std(times_ms),
            'p50_latency_ms': np.percentile(times_ms, 50),
            'p95_latency_ms': np.percentile(times_ms, 95),
            'p99_latency_ms': np.percentile(times_ms, 99),
            'throughput_samples_per_sec': len(test_data) / np.mean(times)
        }


class EdgeDeploymentManager:
    """Manage edge deployment configurations and scripts."""
    
    def __init__(self, config: DictConfig):
        """Initialize deployment manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.deployment_configs = {}
    
    def create_deployment_config(
        self,
        device_type: str,
        model_format: str,
        model_path: str
    ) -> Dict[str, Any]:
        """Create deployment configuration for specific device.
        
        Args:
            device_type: Target device type
            model_format: Model format (onnx, tflite, coreml)
            model_path: Path to model file
            
        Returns:
            Deployment configuration
        """
        device_config = self.config.device.get(device_type, {})
        
        deployment_config = {
            'device_type': device_type,
            'model_format': model_format,
            'model_path': model_path,
            'device_name': device_config.get('device_name', 'Unknown'),
            'cpu_cores': device_config.get('cpu_cores', 1),
            'memory_gb': device_config.get('memory_gb', 1),
            'power_consumption_w': device_config.get('power_consumption_w', 1.0),
            'inference_target': device_config.get('inference_target', 'cpu'),
            'supported_formats': device_config.get('supported_formats', []),
            'optimization_flags': device_config.get('optimization_flags', []),
            'max_batch_size': device_config.get('max_batch_size', 1),
            'max_sequence_length': device_config.get('max_sequence_length', 5),
            'deployment_timestamp': time.time()
        }
        
        self.deployment_configs[device_type] = deployment_config
        return deployment_config
    
    def generate_deployment_script(
        self,
        device_type: str,
        output_path: Union[str, Path]
    ) -> str:
        """Generate deployment script for specific device.
        
        Args:
            device_type: Target device type
            output_path: Output path for script
            
        Returns:
            Path to generated script
        """
        if device_type not in self.deployment_configs:
            raise ValueError(f"No deployment config found for device: {device_type}")
        
        config = self.deployment_configs[device_type]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate script content based on device type
        if device_type == 'raspberry_pi':
            script_content = self._generate_raspberry_pi_script(config)
        elif device_type == 'jetson_nano':
            script_content = self._generate_jetson_nano_script(config)
        elif device_type == 'android':
            script_content = self._generate_android_script(config)
        elif device_type == 'ios':
            script_content = self._generate_ios_script(config)
        else:
            script_content = self._generate_generic_script(config)
        
        # Write script
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Deployment script generated: {output_path}")
        return str(output_path)
    
    def _generate_raspberry_pi_script(self, config: Dict[str, Any]) -> str:
        """Generate Raspberry Pi deployment script."""
        return f"""#!/bin/bash
# Raspberry Pi Deployment Script for Edge NLP Applications
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "Setting up Edge NLP Application on Raspberry Pi..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv edge_nlp_env
source edge_nlp_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install tensorflow-lite-runtime
pip install onnxruntime

# Create application directory
mkdir -p /opt/edge_nlp_app
cd /opt/edge_nlp_app

# Copy model file
cp {config['model_path']} ./model.{config['model_format']}

# Create systemd service
cat > /etc/systemd/system/edge-nlp-app.service << EOF
[Unit]
Description=Edge NLP Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/edge_nlp_app
ExecStart=/opt/edge_nlp_app/edge_nlp_env/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable edge-nlp-app
sudo systemctl start edge-nlp-app

echo "Edge NLP Application deployed successfully!"
echo "Service status: sudo systemctl status edge-nlp-app"
echo "Logs: sudo journalctl -u edge-nlp-app -f"
"""
    
    def _generate_jetson_nano_script(self, config: Dict[str, Any]) -> str:
        """Generate Jetson Nano deployment script."""
        return f"""#!/bin/bash
# Jetson Nano Deployment Script for Edge NLP Applications
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "Setting up Edge NLP Application on Jetson Nano..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv edge_nlp_env
source edge_nlp_env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install tensorflow-lite-runtime
pip install onnxruntime-gpu

# Create application directory
mkdir -p /opt/edge_nlp_app
cd /opt/edge_nlp_app

# Copy model file
cp {config['model_path']} ./model.{config['model_format']}

# Set GPU memory fraction
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Create systemd service
cat > /etc/systemd/system/edge-nlp-app.service << EOF
[Unit]
Description=Edge NLP Application
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/opt/edge_nlp_app
Environment=TF_FORCE_GPU_ALLOW_GROWTH=true
ExecStart=/opt/edge_nlp_app/edge_nlp_env/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable edge-nlp-app
sudo systemctl start edge-nlp-app

echo "Edge NLP Application deployed successfully!"
echo "Service status: sudo systemctl status edge-nlp-app"
echo "Logs: sudo journalctl -u edge-nlp-app -f"
"""
    
    def _generate_android_script(self, config: Dict[str, Any]) -> str:
        """Generate Android deployment script."""
        return f"""#!/bin/bash
# Android Deployment Script for Edge NLP Applications
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

echo "Android deployment requires Android Studio and NDK setup"
echo "Model format: {config['model_format']}"
echo "Model path: {config['model_path']}"

# Create Android project structure
mkdir -p android_app/app/src/main/assets
mkdir -p android_app/app/src/main/java/com/edgenlp/app

# Copy model to assets
cp {config['model_path']} android_app/app/src/main/assets/model.{config['model_format']}

echo "Android project structure created"
echo "Next steps:"
echo "1. Import project in Android Studio"
echo "2. Add TensorFlow Lite or ONNX Runtime dependencies"
echo "3. Implement inference code"
echo "4. Build and deploy to device"
"""
    
    def _generate_ios_script(self, config: Dict[str, Any]) -> str:
        """Generate iOS deployment script."""
        return f"""#!/bin/bash
# iOS Deployment Script for Edge NLP Applications
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

echo "iOS deployment requires Xcode and CoreML setup"
echo "Model format: {config['model_format']}"
echo "Model path: {config['model_path']}"

# Create iOS project structure
mkdir -p ios_app/EdgeNLPApp/EdgeNLPApp
mkdir -p ios_app/EdgeNLPApp/EdgeNLPApp/Models

# Copy model to project
cp {config['model_path']} ios_app/EdgeNLPApp/EdgeNLPApp/Models/model.{config['model_format']}

echo "iOS project structure created"
echo "Next steps:"
echo "1. Open project in Xcode"
echo "2. Add CoreML or ONNX Runtime framework"
echo "3. Implement inference code"
echo "4. Build and deploy to device"
"""
    
    def _generate_generic_script(self, config: Dict[str, Any]) -> str:
        """Generate generic deployment script."""
        return f"""#!/bin/bash
# Generic Edge Device Deployment Script for Edge NLP Applications
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "Setting up Edge NLP Application on generic edge device..."

# Install dependencies
pip install --upgrade pip
pip install numpy pandas scikit-learn
pip install tensorflow-lite-runtime
pip install onnxruntime

# Create application directory
mkdir -p /opt/edge_nlp_app
cd /opt/edge_nlp_app

# Copy model file
cp {config['model_path']} ./model.{config['model_format']}

echo "Edge NLP Application deployed successfully!"
echo "Run: python app.py"
"""
