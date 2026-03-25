
import argparse
import os
import sys
import pickle
import torch
import mlflow
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models.recall.dssm import DSSM
from src.serving.redis_storage import RedisStorage
from src.utils.config_loader import load_config, get_project_root

def generate_full_recall_for_model(model_name: str, run_id: str):
    """
    Generates top-K recall results for all users using a specific trained model
    and stores them in Redis.

    Args:
        model_name (str): The name of the model type (e.g., 'dssm_pointwise').
        run_id (str): The MLflow run ID from which to load the model.
    """
    print(f"--- Starting Full Recall Generation for {model_name} (Run ID: {run_id}) ---")

    # --- 1. Load Configurations and Metadata ---
    root_dir = get_project_root()
    data_config = load_config(os.path.join(root_dir, 'config/data.yaml'))
    processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])
    meta_file_path = os.path.join(processed_dir, data_config['meta_file'])
    print(f"Loading metadata from: {meta_file_path}")
    with open(meta_file_path, 'rb') as f:
        meta_data = pickle.load(f)

    # --- 2. Load Model from MLflow Artifacts ---
    print(f"Loading model from MLflow run: {run_id}")
    artifact_path = "model/dssm_in_batch_best.pth"
    if "pointwise" in model_name:
        artifact_path = "model/dssm_pointwise_best.pth"

    # Load model configuration used during training
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data
    params = run_data.params
    
    model = DSSM(
        total_vocab_size=meta_data['feature_dims'],
        embedding_dim=int(params['embedding_dim']),
        hidden_dims=[int(d) for d in params['hidden_dims'].strip('[]').split(', ')],
        user_feature_count=len(meta_data['user_feature_cols']),
        item_feature_count=len(meta_data['item_feature_cols'])
    )

    # Load the state dict from the artifact path
    model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")

    device = torch.device("cpu") # Use CPU for inference job
    model.to(device)
    model.eval()

    # --- 3. Load All User/Item Features and Interaction History ---
    print("Loading all user/item data for inference...")
    # Construct the correct paths to the .dat files
    raw_data_dir = os.path.join(root_dir, data_config['raw_data_dir'])
    user_feature_file = os.path.join(raw_data_dir, data_config['user_feature_file'])
    item_feature_file = os.path.join(raw_data_dir, data_config['item_feature_file'])

    print(f"Loading user features from: {user_feature_file}")
    user_features_df = pd.read_csv(user_feature_file, sep=',', header=None, names=meta_data['user_feature_cols'], engine='python')
    print(f"Loading item features from: {item_feature_file}")
    item_features_df = pd.read_csv(item_feature_file, sep=',', header=None, names=meta_data['item_feature_cols'], engine='python')
    train_file_path = os.path.join(processed_dir, data_config['train_file'])
    print(f"Loading training history from: {train_file_path}")
    train_df = pd.read_csv(train_file_path, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Create a map of user_id -> seen item_ids
    user_history = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    # --- 4. Generate All Item Embeddings ---
    print("Generating all item embeddings...")
    # Clip IDs to be within the range of the embedding matrix
    feature_cols = meta_data['item_feature_cols']
    max_id = meta_data['feature_dims'] - 1
    item_features_df[feature_cols] = item_features_df[feature_cols].clip(upper=max_id)

    item_tensors = torch.LongTensor(item_features_df[meta_data['item_feature_cols']].values.astype(np.int64)).to(device)
    all_item_vectors = model.get_item_vector(item_tensors)

    # --- 5. Generate Recall for Each User and Store in Redis ---
    print("Connecting to Redis (db=5)...")
    redis_storage = RedisStorage(db=5)
    redis_storage.client.ping() # Check connection
    print("Redis connection successful.")

    print(f"Generating recall and writing to Redis for {len(user_features_df)} users...")
    
    recall_key_prefix = f"recall:{model_name}"
    
    with torch.no_grad():
        for _, user_row in tqdm(user_features_df.iterrows(), total=len(user_features_df)):
            user_id = user_row['user_id']

            # Clip user feature IDs
            user_feature_cols = meta_data['user_feature_cols']
            user_row[user_feature_cols] = user_row[user_feature_cols].clip(upper=max_id)

            user_tensor = torch.LongTensor(user_row[meta_data['user_feature_cols']].values.astype(np.int64)).unsqueeze(0).to(device)

            # Generate user vector
            user_vector = model.get_user_vector(user_tensor)

            # Calculate scores against all items
            scores = torch.matmul(user_vector, all_item_vectors.t()).squeeze()

            # Exclude items the user has already seen
            seen_items = user_history.get(user_id, set())
            item_ids_series = item_features_df['item_id']
            # Create a boolean mask for seen items
            seen_mask = item_ids_series.isin(seen_items)
            scores[seen_mask.values] = -float('inf')

            # Get top 100 item IDs
            _, top_indices = torch.topk(scores, k=100)
            top_item_ids = item_ids_series.iloc[top_indices.cpu().numpy()].tolist()

            # Store in Redis
            redis_key = f"{recall_key_prefix}:{user_id}"
            redis_storage.set_recall_list(redis_key, top_item_ids)

    print(f"--- Finished Full Recall Generation for {model_name} ---")

def main():
    parser = argparse.ArgumentParser(description="Run offline recall generation job.")
    parser.add_argument(
        "--model-name", 
        type=str, 
        required=True, 
        help="The name for this recall set (e.g., 'dssm_pointwise')."
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        required=True, 
        help="The MLflow Run ID of the trained model to use."
    )
    args = parser.parse_args()

    generate_full_recall_for_model(args.model_name, args.run_id)

if __name__ == "__main__":
    main()
