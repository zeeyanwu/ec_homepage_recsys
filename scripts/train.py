
import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.pipeline.train_pipeline import run_train_pipeline

def main():
    """
    Main entry point for the training pipeline.
    Parses command-line arguments to get the model configuration file,
    loads the configuration, and runs the training pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the training pipeline for a specified model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file.')
    
    args = parser.parse_args()
    
    # Load model-specific configuration
    config = load_config(args.config)
    
    # Run the training pipeline
    run_train_pipeline(config)

if __name__ == "__main__":
    main()
