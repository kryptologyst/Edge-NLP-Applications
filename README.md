# Edge NLP Applications

A comprehensive framework for deploying natural language processing models on edge devices, enabling real-time intent recognition for voice commands and text inputs without cloud dependency.

## Overview

This project demonstrates how to build, optimize, and deploy lightweight NLP models for edge computing scenarios. The system provides a complete pipeline from data preparation and model training to edge deployment and performance monitoring.

## Key Features

- **Multi-Framework Support**: PyTorch and TensorFlow implementations
- **Edge Optimization**: Model quantization, pruning, and compression techniques
- **Multiple Deployment Targets**: Raspberry Pi, Jetson Nano, Android, iOS
- **Real-time Inference**: Low-latency prediction optimized for edge constraints
- **Comprehensive Evaluation**: Accuracy and performance benchmarking
- **Interactive Demo**: Streamlit-based web interface for testing and visualization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Model Training │    │ Edge Deployment │
│                 │    │                 │    │                 │
│ • Synthetic Data│───▶│ • LSTM Models   │───▶│ • TFLite/ONNX   │
│ • Tokenization  │    │ • Quantization  │    │ • CoreML        │
│ • Preprocessing │    │ • Compression   │    │ • Device Scripts│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Evaluation    │    │   Benchmarking  │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Accuracy      │    │ • Latency       │    │ • Performance   │
│ • Confusion     │    │ • Throughput    │    │ • Resource Usage│
│ • Per-class     │    │ • Memory        │    │ • Edge Metrics  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Edge-NLP-Applications.git
   cd Edge-NLP-Applications
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive demo**
   ```bash
   streamlit run demo.py
   ```

4. **Train a model**
   ```bash
   python train.py --framework tensorflow --export --benchmark
   ```

## Project Structure

```
0798_Edge_NLP_Applications/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── intent_classifier.py   # LSTM-based intent classifier
│   ├── pipelines/                 # Training pipelines
│   │   └── training.py           # Training pipeline
│   ├── export/                    # Export and deployment
│   │   └── deployment.py         # Model export utilities
│   ├── utils/                     # Utility functions
│   │   ├── core.py               # Core utilities
│   │   ├── data_utils.py         # Data processing
│   │   └── evaluation.py         # Evaluation metrics
│   └── runtimes/                 # Runtime implementations
├── configs/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   ├── device/                   # Device-specific configs
│   └── quant/                    # Quantization configs
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   └── processed/               # Processed data
├── scripts/                      # Utility scripts
├── tests/                        # Test files
├── assets/                       # Generated assets
├── demo/                         # Demo files
├── train.py                      # Main training script
├── demo.py                       # Streamlit demo
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Usage

### Training Models

Train a model with default settings:
```bash
python train.py
```

Train with specific framework and export options:
```bash
python train.py --framework pytorch --export --benchmark --output-dir outputs
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo.py
```

The demo provides:
- Real-time intent recognition
- Model performance analysis
- Edge benchmarking tools
- Training visualization
- Interactive testing interface

### Configuration

Modify `configs/config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Device-specific settings
- Quantization options

## Model Architecture

The system implements lightweight LSTM-based intent classifiers:

- **Input**: Tokenized text sequences (max length: 5 tokens)
- **Embedding**: Word embeddings (16-64 dimensions)
- **LSTM**: Bidirectional LSTM (32-128 hidden units)
- **Output**: Softmax classification over intent classes
- **Optimization**: Dropout, batch normalization, early stopping

## Edge Optimization Techniques

### Quantization
- **Post-Training Quantization (PTQ)**: Convert FP32 to INT8
- **Quantization-Aware Training (QAT)**: Train with quantization simulation
- **Dynamic Quantization**: Runtime quantization for PyTorch models

### Model Compression
- **Pruning**: Magnitude-based weight pruning
- **Knowledge Distillation**: Teacher-student model compression
- **Architecture Search**: Hardware-aware neural architecture search

### Deployment Formats
- **TensorFlow Lite**: Android, Raspberry Pi, microcontrollers
- **ONNX Runtime**: Cross-platform inference
- **CoreML**: iOS and macOS deployment
- **OpenVINO**: Intel hardware acceleration

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | > 85% | ~90% |
| Latency | < 50ms | ~25ms |
| Model Size | < 1MB | ~0.5MB |
| Memory Usage | < 10MB | ~5MB |
| Throughput | > 100 samples/sec | ~200 samples/sec |

## Device Support

### Raspberry Pi 4B
- **CPU**: ARM Cortex-A72 (4 cores)
- **Memory**: 4GB RAM
- **Power**: 3.4W
- **Formats**: TensorFlow Lite, ONNX Runtime

### Jetson Nano
- **CPU**: ARM Cortex-A57 (4 cores)
- **GPU**: NVIDIA Maxwell (128 CUDA cores)
- **Memory**: 4GB RAM
- **Power**: 5W
- **Formats**: TensorFlow Lite, ONNX Runtime, TensorRT

### Android Devices
- **CPU**: ARM/x86 (varies)
- **Memory**: 2-8GB RAM
- **Power**: 1-3W
- **Formats**: TensorFlow Lite, ONNX Runtime

### iOS Devices
- **CPU**: Apple A-series (varies)
- **Memory**: 2-6GB RAM
- **Power**: 1-2W
- **Formats**: CoreML, ONNX Runtime

## Evaluation Metrics

### Model Quality
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted metrics
- **Confusion Matrix**: Detailed classification analysis
- **ROC/AUC**: Receiver operating characteristic analysis

### Edge Performance
- **Latency**: Inference time (mean, p50, p95, p99)
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak RAM consumption
- **Model Size**: Compressed model file size
- **Energy Consumption**: Power usage per inference

### Robustness
- **Noise Tolerance**: Performance under noisy conditions
- **Edge Cases**: Handling of out-of-vocabulary inputs
- **Offline Operation**: Functionality without network connectivity

## Safety and Privacy

### Privacy Protection
- **On-Device Processing**: No data leaves the device
- **No Cloud Dependency**: Fully offline operation
- **Data Minimization**: Only necessary data is processed
- **Secure Storage**: Encrypted model files

### Safety Measures
- **Input Validation**: Sanitization of user inputs
- **Error Handling**: Graceful failure modes
- **Resource Limits**: Memory and CPU usage bounds
- **Audit Logging**: Comprehensive operation logs

## Limitations and Disclaimers

### Current Limitations
- **Synthetic Data**: Models trained on generated datasets
- **Limited Vocabulary**: Small vocabulary size for edge constraints
- **Simple Intents**: Basic intent classification only
- **No Context**: No conversation history or context awareness

### Important Disclaimers

⚠️ **NOT FOR SAFETY-CRITICAL USE**

This software is intended for research and educational purposes only. It is NOT suitable for:
- Safety-critical applications
- Production deployment without proper validation
- Medical or automotive systems
- Security-sensitive applications

**Use at your own risk.** No warranty or guarantee of accuracy, reliability, or safety is provided.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{edge_nlp_applications,
  title={Edge NLP Applications},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Edge-NLP-Applications}
}
```

## Acknowledgments

- TensorFlow and PyTorch communities for excellent ML frameworks
- Edge computing research community
- Open source contributors and maintainers

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the demo application
- Contact the development team

---

**Remember**: This is a research and educational project. Always validate thoroughly before any production use.
# Edge-NLP-Applications
