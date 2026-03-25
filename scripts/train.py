
import argparse
import sys
import os

# Add the project root to the Python path to resolve module imports.
# This makes the script runnable from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.train_pipeline import run_train_pipeline

def main():
    """Parses command-line arguments and triggers the training pipeline."""
    parser = argparse.ArgumentParser(description="Run a training pipeline for a recommendation model.")
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the model configuration YAML file (e.g., config/dssm_pointwise.yaml).'
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the MLflow run. If not provided, it's derived from the config file name."
    )
    
    args = parser.parse_args()

    # If run_name is not provided, create a default name from the config file name
    run_name = args.run_name
    if not run_name:
        run_name = os.path.splitext(os.path.basename(args.config))[0]

    print(f"Starting training run: '{run_name}' with config: '{args.config}'")
    run_train_pipeline(config_path=args.config, run_name=run_name)

if __name__ == "__main__":
    main()
