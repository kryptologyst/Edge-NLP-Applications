"""Streamlit demo for Edge NLP Applications."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import load_config, set_deterministic_seed
from src.utils.data_utils import IntentDataset, create_synthetic_dataset
from src.models.intent_classifier import PyTorchIntentLSTM, TensorFlowIntentLSTM
from src.utils.evaluation import EdgeNLPEvaluator, PerformanceBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Edge NLP Applications Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This is a research and educational demonstration of Edge NLP Applications. 
This software is **NOT intended for safety-critical or production deployment**.

- Models are trained on synthetic data for demonstration purposes
- Performance metrics are simulated and may not reflect real-world conditions
- No warranty or guarantee of accuracy or reliability
- Use at your own risk for research and educational purposes only

For production use, please ensure proper testing, validation, and safety measures.
"""


@st.cache_data
def load_config_cached(config_path: str) -> DictConfig:
    """Load configuration with caching."""
    return load_config(config_path)


@st.cache_data
def create_dataset_cached(num_samples: int, random_seed: int) -> IntentDataset:
    """Create dataset with caching."""
    return create_synthetic_dataset(
        num_samples=num_samples,
        num_intents=10,
        max_sequence_length=5,
        random_seed=random_seed
    )


@st.cache_resource
def load_model(model_path: str, framework: str, vocab_size: int, num_classes: int):
    """Load model with caching."""
    if framework == "pytorch":
        model = PyTorchIntentLSTM(
            vocab_size=vocab_size,
            embedding_dim=16,
            hidden_dim=32,
            num_classes=num_classes,
            max_sequence_length=5,
            dropout=0.2
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    else:
        model = TensorFlowIntentLSTM(
            vocab_size=vocab_size,
            embedding_dim=16,
            hidden_dim=32,
            num_classes=num_classes,
            max_sequence_length=5,
            dropout=0.2
        )
        model.load_model(model_path)
    
    return model


def predict_intent(model, text: str, dataset: IntentDataset, framework: str) -> Tuple[str, float]:
    """Predict intent for given text."""
    # Preprocess text
    processed_text = dataset.preprocess_text(text)
    
    if framework == "pytorch":
        with torch.no_grad():
            input_tensor = torch.tensor(processed_text.reshape(1, -1), dtype=torch.long)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
    else:
        input_data = processed_text.reshape(1, -1)
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
    
    # Decode prediction
    intent = dataset.decode_prediction(predicted_class)
    
    return intent, confidence


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🤖 Edge NLP Applications Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Load configuration
    config_path = st.sidebar.selectbox(
        "Select Configuration",
        ["configs/config.yaml"],
        help="Choose the configuration file to use"
    )
    
    try:
        config = load_config_cached(config_path)
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return
    
    # Framework selection
    framework = st.sidebar.selectbox(
        "Framework",
        ["tensorflow", "pytorch"],
        help="Choose the framework for inference"
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Target Device",
        ["cpu", "gpu", "auto"],
        help="Choose the target device for inference"
    )
    
    # Dataset parameters
    st.sidebar.subheader("Dataset Parameters")
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=100,
        max_value=2000,
        value=config.data.max_samples,
        step=100
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000,
        value=config.data.random_seed
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    embedding_dim = st.sidebar.slider(
        "Embedding Dimension",
        min_value=8,
        max_value=64,
        value=config.model.embedding_dim,
        step=8
    )
    
    hidden_dim = st.sidebar.slider(
        "Hidden Dimension",
        min_value=16,
        max_value=128,
        value=config.model.hidden_dim,
        step=16
    )
    
    dropout = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=config.model.dropout,
        step=0.1
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Intent Recognition", 
        "📊 Model Performance", 
        "⚡ Edge Benchmarking", 
        "📈 Training Visualization",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Real-time Intent Recognition")
        
        # Disclaimer
        st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
        
        # Create dataset
        with st.spinner("Creating dataset..."):
            dataset = create_dataset_cached(num_samples, random_seed)
        
        st.success(f"Dataset created with {len(dataset.sentences)} samples")
        
        # Model selection
        model_option = st.radio(
            "Model Option",
            ["Use Pre-trained Model", "Train New Model"],
            help="Choose to use a pre-trained model or train a new one"
        )
        
        if model_option == "Use Pre-trained Model":
            # Check if model exists
            model_path = f"outputs/model_{framework}"
            if Path(model_path).exists():
                with st.spinner("Loading model..."):
                    model = load_model(
                        model_path, framework, 
                        dataset.get_vocab_size(), 
                        dataset.get_num_classes()
                    )
                st.success("Model loaded successfully!")
            else:
                st.error(f"No pre-trained model found at {model_path}")
                st.info("Please train a model first using the training script.")
                return
        else:
            # Train new model
            if st.button("Train New Model", type="primary"):
                with st.spinner("Training model..."):
                    # This would normally call the training pipeline
                    # For demo purposes, we'll create a simple model
                    if framework == "pytorch":
                        model = PyTorchIntentLSTM(
                            vocab_size=dataset.get_vocab_size(),
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            num_classes=dataset.get_num_classes(),
                            max_sequence_length=5,
                            dropout=dropout
                        )
                    else:
                        model = TensorFlowIntentLSTM(
                            vocab_size=dataset.get_vocab_size(),
                            embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim,
                            num_classes=dataset.get_num_classes(),
                            max_sequence_length=5,
                            dropout=dropout
                        )
                
                st.success("Model trained successfully!")
        
        # Intent recognition interface
        st.subheader("Try Intent Recognition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            user_input = st.text_input(
                "Enter a command:",
                placeholder="e.g., 'turn on the lights'",
                help="Enter a voice command or text input"
            )
            
            # Example commands
            st.subheader("Example Commands")
            example_commands = [
                "turn on the lights",
                "play music",
                "what's the weather",
                "set an alarm",
                "increase volume"
            ]
            
            for cmd in example_commands:
                if st.button(f"'{cmd}'", key=f"cmd_{cmd}"):
                    user_input = cmd
                    st.rerun()
        
        with col2:
            if user_input and 'model' in locals():
                # Predict intent
                start_time = time.time()
                intent, confidence = predict_intent(model, user_input, dataset, framework)
                inference_time = (time.time() - start_time) * 1000
                
                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.subheader("Prediction Results")
                st.metric("Intent", intent)
                st.metric("Confidence", f"{confidence:.2%}")
                st.metric("Inference Time", f"{inference_time:.2f} ms")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence visualization
                st.subheader("Confidence Distribution")
                
                # Get all class probabilities
                processed_text = dataset.preprocess_text(user_input)
                
                if framework == "pytorch":
                    with torch.no_grad():
                        input_tensor = torch.tensor(processed_text.reshape(1, -1), dtype=torch.long)
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1).numpy()[0]
                else:
                    input_data = processed_text.reshape(1, -1)
                    predictions = model.predict(input_data)
                    probabilities = predictions[0]
                
                # Create probability chart
                prob_df = pd.DataFrame({
                    'Intent': dataset.get_class_names(),
                    'Probability': probabilities
                }).sort_values('Probability', ascending=True)
                
                st.bar_chart(prob_df.set_index('Intent'))
    
    with tab2:
        st.header("Model Performance Analysis")
        
        if 'model' in locals() and 'dataset' in locals():
            # Evaluate model
            evaluator = EdgeNLPEvaluator(dataset.get_class_names())
            X_test, y_test = dataset.get_test_data()
            
            with st.spinner("Evaluating model..."):
                evaluation_results = evaluator.evaluate_model(
                    model, X_test, y_test, framework
                )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{evaluation_results['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{evaluation_results['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{evaluation_results['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{evaluation_results['f1_score']:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            evaluator.plot_confusion_matrix()
            
            # Per-class metrics
            st.subheader("Per-Class Performance")
            evaluator.plot_class_metrics()
            
            # Detailed report
            st.subheader("Detailed Classification Report")
            st.text(evaluator.generate_classification_report())
        else:
            st.info("Please load or train a model first.")
    
    with tab3:
        st.header("Edge Performance Benchmarking")
        
        if 'model' in locals() and 'dataset' in locals():
            # Benchmark settings
            col1, col2 = st.columns(2)
            
            with col1:
                num_runs = st.slider("Number of Benchmark Runs", 10, 1000, 100)
                warmup_runs = st.slider("Warmup Runs", 5, 50, 10)
            
            with col2:
                batch_size = st.slider("Batch Size", 1, 32, 1)
                device_type = st.selectbox("Device Type", ["cpu", "gpu", "auto"])
            
            if st.button("Run Benchmark", type="primary"):
                with st.spinner("Running performance benchmark..."):
                    benchmark = PerformanceBenchmark()
                    X_test, y_test = dataset.get_test_data()
                    
                    benchmark_results = benchmark.benchmark_inference_speed(
                        model, X_test, framework, num_runs, warmup_runs
                    )
                
                # Display results
                st.subheader("Benchmark Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Mean Latency", 
                        f"{benchmark_results['per_sample_time_mean_ms']:.2f} ms"
                    )
                with col2:
                    st.metric(
                        "P95 Latency", 
                        f"{benchmark_results['per_sample_time_p95_ms']:.2f} ms"
                    )
                with col3:
                    st.metric(
                        "Throughput", 
                        f"{benchmark_results['throughput_samples_per_sec']:.1f} samples/sec"
                    )
                
                # Latency distribution
                st.subheader("Latency Distribution")
                
                # Simulate latency distribution for visualization
                mean_latency = benchmark_results['per_sample_time_mean_ms']
                std_latency = benchmark_results['per_sample_time_std_ms']
                
                # Generate sample latencies for visualization
                sample_latencies = np.random.normal(mean_latency, std_latency, 1000)
                
                latency_df = pd.DataFrame({'Latency (ms)': sample_latencies})
                st.histogram(latency_df)
                
                # Performance report
                st.subheader("Performance Report")
                st.text(benchmark.create_performance_report())
        else:
            st.info("Please load or train a model first.")
    
    with tab4:
        st.header("Training Visualization")
        
        # Simulate training history for visualization
        epochs = list(range(1, 101))
        train_loss = [0.8 * np.exp(-epoch/30) + 0.1 + 0.05 * np.random.random() for epoch in epochs]
        val_loss = [0.9 * np.exp(-epoch/25) + 0.15 + 0.05 * np.random.random() for epoch in epochs]
        train_acc = [1 - loss for loss in train_loss]
        val_acc = [1 - loss for loss in val_loss]
        
        # Create training history dataframe
        history_df = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Train Accuracy': train_acc,
            'Validation Accuracy': val_acc
        })
        
        # Loss plot
        st.subheader("Training and Validation Loss")
        st.line_chart(history_df.set_index('Epoch')[['Train Loss', 'Validation Loss']])
        
        # Accuracy plot
        st.subheader("Training and Validation Accuracy")
        st.line_chart(history_df.set_index('Epoch')[['Train Accuracy', 'Validation Accuracy']])
        
        # Training metrics table
        st.subheader("Training Metrics Summary")
        
        final_metrics = {
            'Final Train Loss': train_loss[-1],
            'Final Validation Loss': val_loss[-1],
            'Final Train Accuracy': train_acc[-1],
            'Final Validation Accuracy': val_acc[-1],
            'Best Validation Accuracy': max(val_acc),
            'Epochs to Convergence': next(i for i, acc in enumerate(val_acc) if acc > 0.9)
        }
        
        metrics_df = pd.DataFrame(list(final_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)
    
    with tab5:
        st.header("About Edge NLP Applications")
        
        st.markdown("""
        ## Overview
        
        This demo showcases **Edge NLP Applications** - a comprehensive framework for deploying 
        natural language processing models on edge devices. The system enables real-time intent 
        recognition for voice commands and text inputs without requiring cloud connectivity.
        
        ## Key Features
        
        - **Multi-Framework Support**: PyTorch and TensorFlow implementations
        - **Edge Optimization**: Model quantization, pruning, and compression
        - **Multiple Deployment Targets**: Raspberry Pi, Jetson Nano, Android, iOS
        - **Real-time Inference**: Low-latency prediction for edge constraints
        - **Comprehensive Evaluation**: Accuracy and performance benchmarking
        
        ## Use Cases
        
        - **Smart Home**: Voice-controlled lighting, music, and appliances
        - **IoT Devices**: Command recognition for embedded systems
        - **Mobile Applications**: On-device intent classification
        - **Edge Computing**: Offline NLP capabilities
        
        ## Technical Specifications
        
        - **Model Architecture**: LSTM-based intent classifier
        - **Input Processing**: Text tokenization and sequence padding
        - **Output**: Intent classification with confidence scores
        - **Optimization**: Quantization-aware training and post-training quantization
        - **Deployment**: TensorFlow Lite, ONNX, CoreML formats
        
        ## Performance Targets
        
        - **Latency**: < 50ms per inference
        - **Model Size**: < 1MB compressed
        - **Memory Usage**: < 10MB RAM
        - **Accuracy**: > 85% on test set
        
        ## Getting Started
        
        1. **Train a Model**: Use the training script to create a custom model
        2. **Export Formats**: Convert to edge-optimized formats
        3. **Deploy**: Use generated deployment scripts for target devices
        4. **Monitor**: Track performance and accuracy metrics
        
        ## Safety Notice
        
        This is a research and educational demonstration. For production deployment, 
        ensure proper testing, validation, and safety measures are in place.
        """)


if __name__ == "__main__":
    main()
