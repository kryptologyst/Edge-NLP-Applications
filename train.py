#!/usr/bin/env python3
"""Main training script for Edge NLP Applications."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.core import setup_logging, set_deterministic_seed, load_config
from src.utils.data_utils import create_synthetic_dataset, save_dataset_to_file
from src.pipelines.training import TrainingPipeline
from src.utils.evaluation import EdgeNLPEvaluator, PerformanceBenchmark, create_leaderboard
from src.export.deployment import ModelExporter, EdgeDeploymentManager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Edge NLP Applications Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--framework", 
        type=str, 
        choices=["pytorch", "tensorflow"],
        default="tensorflow",
        help="Framework to use for training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use for training"
    )
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export model to edge formats after training"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Output directory for models and results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device != "auto":
        config.deployment.target_device = args.device
    
    # Setup logging
    logger = setup_logging(
        level=config.safety.log_level,
        log_file=f"{args.output_dir}/training.log"
    )
    
    logger.info("Starting Edge NLP Applications Training")
    logger.info(f"Framework: {args.framework}")
    logger.info(f"Device: {config.deployment.target_device}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set random seeds
    set_deterministic_seed(config.data.random_seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create dataset
        logger.info("Creating synthetic dataset...")
        dataset = create_synthetic_dataset(
            num_samples=config.data.max_samples,
            num_intents=10,
            max_sequence_length=config.model.max_sequence_length,
            random_seed=config.data.random_seed
        )
        
        # Save dataset
        save_dataset_to_file(dataset, output_dir / "dataset.json")
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(config)
        pipeline.prepare_data(dataset)
        pipeline.create_model(args.framework)
        
        # Train model
        logger.info("Starting model training...")
        if args.framework == "pytorch":
            training_history = pipeline.train_pytorch()
        else:
            training_history = pipeline.train_tensorflow()
        
        # Save training history
        import json
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = EdgeNLPEvaluator(dataset.get_class_names())
        X_test, y_test = dataset.get_test_data()
        
        evaluation_results = evaluator.evaluate_model(
            pipeline.model, X_test, y_test, args.framework
        )
        
        # Save evaluation results
        evaluator.save_results(output_dir / "evaluation_results.json")
        
        # Generate plots
        evaluator.plot_confusion_matrix(output_dir / "confusion_matrix.png")
        evaluator.plot_class_metrics(output_dir / "class_metrics.png")
        
        # Print evaluation report
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(evaluator.generate_classification_report())
        
        # Save model
        model_path = output_dir / f"model_{args.framework}"
        pipeline.save_model(model_path)
        
        # Export to edge formats
        if args.export:
            logger.info("Exporting model to edge formats...")
            exporter = ModelExporter(config)
            
            # Create sample input for export
            sample_input = X_test[:1]
            
            if args.framework == "pytorch":
                import torch
                sample_tensor = torch.tensor(sample_input, dtype=torch.long)
                exporter.export_pytorch_to_onnx(
                    pipeline.model, sample_tensor, output_dir / "model.onnx"
                )
            else:
                exporter.export_tensorflow_to_tflite(
                    pipeline.model, output_dir / "model.tflite",
                    quantization=config.deployment.quantization,
                    representative_dataset=X_test[:100]
                )
                
                # Also export to CoreML for iOS
                try:
                    exporter.export_to_coreml(
                        pipeline.model, output_dir / "model.mlmodel"
                    )
                except ImportError:
                    logger.warning("CoreML export skipped (coremltools not available)")
        
        # Run benchmarks
        if args.benchmark:
            logger.info("Running performance benchmarks...")
            benchmark = PerformanceBenchmark()
            
            benchmark_results = benchmark.benchmark_inference_speed(
                pipeline.model, X_test, args.framework
            )
            
            benchmark.save_benchmark_results(output_dir / "benchmark_results.json")
            
            print("\n" + "="*60)
            print("PERFORMANCE BENCHMARK")
            print("="*60)
            print(benchmark.create_performance_report())
        
        # Create leaderboard
        model_info = pipeline.get_model_info()
        model_info.update(evaluation_results)
        model_info['framework'] = args.framework
        
        leaderboard = create_leaderboard([model_info], output_dir / "leaderboard.csv")
        
        print("\n" + "="*60)
        print("LEADERBOARD")
        print("="*60)
        print(leaderboard.to_string(index=False))
        
        # Generate deployment scripts
        if args.export:
            logger.info("Generating deployment scripts...")
            deployment_manager = EdgeDeploymentManager(config)
            
            # Create deployment configs for different devices
            devices = ["raspberry_pi", "jetson_nano", "android", "ios"]
            
            for device in devices:
                model_format = "onnx" if args.framework == "pytorch" else "tflite"
                model_file = output_dir / f"model.{model_format}"
                
                if model_file.exists():
                    deployment_config = deployment_manager.create_deployment_config(
                        device, model_format, str(model_file)
                    )
                    
                    script_path = output_dir / f"deploy_{device}.sh"
                    deployment_manager.generate_deployment_script(device, script_path)
                    
                    # Make script executable
                    script_path.chmod(0o755)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Framework: {args.framework}")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
        print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
        print(f"Test Samples: {evaluation_results['num_test_samples']}")
        print(f"Output Directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
