import os
import pickle

from src.etl.loader import load_and_merge_data
from src.etl.feature_builder import build_feature_map, transform_data_with_feature_map, compute_and_save_global_score
from src.etl.dataset_splitter import split_and_save_data
from src.utils.config_loader import get_project_root

def run_pipeline(config):
    """
    Orchestrates the entire data processing pipeline.
    """
    print("=== ETL Pipeline Started ===")
    root_dir = get_project_root()

    processed_dir = os.path.join(root_dir, config['processed_data_dir'])
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load and Merge Data
    df = load_and_merge_data(config, root_dir)

    # Define feature columns structure from config (or hardcode for now)
    # This should ideally come from the config file
    user_feature_cols = ['uid', 'utag1', 'utag2']
    item_feature_cols = ['iid', 'itag1', 'itag2', 'itag3']

    # 2. Build Feature Map
    feature_map, feature_dims = build_feature_map(df, user_feature_cols, item_feature_cols)

    # 3. Transform Data with Feature Map
    df_transformed = transform_data_with_feature_map(df, feature_map, user_feature_cols, item_feature_cols)

    # 4. Split and Save Train/Test Data
    train_indices, _ = split_and_save_data(df_transformed, config, root_dir)

    # 5. Compute and Save Global Score
    _ = compute_and_save_global_score(df, train_indices, config, root_dir, feature_map)

    # 6. Save metadata
    meta_data = {
        'feature_dims': feature_dims,
        'user_feature_cols': user_feature_cols,
        'item_feature_cols': item_feature_cols
    }
    meta_path = os.path.join(processed_dir, config['meta_file'])
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)
    print(f"\nMetadata saved to {meta_path}")

    # Save feature map
    feature_map_path = os.path.join(processed_dir, config['feature_map_file'])
    with open(feature_map_path, 'wb') as f:
        pickle.dump(feature_map, f)
    print(f"Feature map saved to {feature_map_path}")

    print("\n=== ETL Pipeline Finished ===")
