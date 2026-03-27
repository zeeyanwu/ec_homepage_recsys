import os
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.serving.redis_storage import RedisStorage
from src.utils.config_loader import load_config, get_project_root

def export_hot_list_to_redis():
    """
    Reads the global item scores and exports them to Redis as a sorted set.
    """
    print("--- Starting Global Hot List Export to Redis ---")

    # --- 1. Load Config and Data ---
    root_dir = get_project_root()
    data_config = load_config(os.path.join(root_dir, 'config/data.yaml'))
    processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])
    
    score_file_path = os.path.join(processed_dir, data_config['global_score_file'])
    print(f"Loading global scores from: {score_file_path}")
    
    try:
        df_score = pd.read_csv(score_file_path)
    except FileNotFoundError:
        print(f"Error: Global score file not found at {score_file_path}.")
        print("Please run the data pipeline first to generate this file.")
        return

    # Create a dictionary of {item_id: score}
    # Ensure iid is string, as redis keys/values will be strings
    item_scores = pd.Series(df_score.global_score.values, index=df_score.iid.astype(str)).to_dict()

    # --- 2. Connect to Redis and Save ---
    print("Connecting to Redis (db=5)...")
    redis_storage = RedisStorage(db=5)
    redis_storage.client.ping() # Check connection
    print("Redis connection successful.")

    print(f"Saving {len(item_scores)} items to the global hot list...")
    redis_storage.save_global_hot_list(item_scores, list_name='global_hot')

    print("--- Finished Global Hot List Export ---")

if __name__ == "__main__":
    export_hot_list_to_redis()
