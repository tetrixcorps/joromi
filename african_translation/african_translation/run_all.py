#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import subprocess

def setup_environment():
    """Set up the Python environment"""
    print("Setting up environment...")
    
    # Install requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install the package in development mode
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Set up NLTK
    subprocess.check_call([sys.executable, "scripts/setup_nltk.py"])

def run_training():
    """Run the training script"""
    print("Starting training...")
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    # Import and run main
    from scripts.finetune_translation import main
    main()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        setup_environment()
        run_training()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
