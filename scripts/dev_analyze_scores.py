import pandas as pd
import os
import sys

# Add project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config

def analyze_scores_cli(config):
    """
    This simplified script reads item_global_score.csv and prints a numerical
    analysis of the key scores directly to the console.
    """
    # Construct the correct file path from the config
    processed_data_dir = os.path.join(PROJECT_ROOT, config['processed_data_dir'])
    file_path = os.path.join(processed_data_dir, config['global_score_file'])

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please run the data pipeline first: python scripts/run_data_pipeline.py --config config/data.yaml")
        return

    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")

    print("\n--- Analysis of 'global_score' ---")
    print(df['global_score'].describe())
    zero_score_percentage = (df['global_score'] == 0).mean() * 100
    print(f"\nPercentage of items with global_score == 0: {zero_score_percentage:.2f}%\n")

    print("--- Analysis of 'ctr_mean' ---")
    print(df['ctr_mean'].describe())
    zero_ctr_percentage = (df['ctr_mean'] == 0).mean() * 100
    print(f"Percentage of items with ctr_mean == 0: {zero_ctr_percentage:.2f}%\n")

    print("--- Analysis of 'freshness_score' ---")
    # Fill potential NaN values before calculation
    df['freshness_score'].fillna(0, inplace=True)
    print(df['freshness_score'].describe())
    zero_decay_percentage = (df['freshness_score'] == 0).mean() * 100
    print(f"Percentage of items with freshness_score == 0: {zero_decay_percentage:.2f}%")

if __name__ == '__main__':
    data_config_path = os.path.join(PROJECT_ROOT, 'config', 'data.yaml')
    if not os.path.exists(data_config_path):
        print(f"Error: Data config file not found at {data_config_path}")
    else:
        config = load_config(data_config_path)
        analyze_scores_cli(config)