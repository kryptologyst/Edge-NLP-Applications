#!/usr/bin/env python3
"""Setup script for Edge NLP Applications."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Edge NLP Applications...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version} detected")
    
    # Install dependencies
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("⚠️ Pre-commit installation failed, continuing...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "outputs",
        "assets",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️ Some tests failed, but setup completed")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the demo: streamlit run demo.py")
    print("2. Train a model: python train.py --framework tensorflow --export")
    print("3. Check the README.md for more information")
    print("\n⚠️ Remember: This is for research/educational use only!")


if __name__ == "__main__":
    main()
