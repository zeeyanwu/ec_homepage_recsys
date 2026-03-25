
import os
import argparse
import pandas as pd
import torch
import json
from tqdm import tqdm

from src.utils.config_loader import get_project_root, load_config
from src.serving.redis_storage import RedisStorage
from src.models.dssm import DSSM

def load_all_data(data_config, meta_data):
    """Loads and combines train/test data to get unique users and items."""
    root_dir = get_project_root()
    processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])
    train_path = os.path.join(processed_dir, data_config['train_file'])
    test_path = os.path.join(processed_dir, data_config['test_file'])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    uid_col = meta_data['user_feature_cols'][0]
    iid_col = meta_data['item_feature_cols'][0]
    
    # Get all unique users with their features
    all_users_df = full_df[meta_data['user_feature_cols']].drop_duplicates(subset=[uid_col]).reset_index(drop=True)
    
    # Get all unique items with their features
    all_items_df = full_df[meta_data['item_feature_cols']].drop_duplicates(subset=[iid_col]).reset_index(drop=True)

    return full_df, all_users_df, all_items_df

def build_user_history(full_data_df, meta_data):
    """Builds a dictionary of items seen by each user from the full dataset."""
    uid_col = meta_data['user_feature_cols'][0]
    iid_col = meta_data['item_feature_cols'][0]
    
    # History contains items with positive interaction
    user_history = full_data_df[full_data_df['label'] > 0].groupby(uid_col)[iid_col].apply(set).to_dict()
    print(f"Built user history for {len(user_history)} users.")
    return user_history

def generate_recalls_for_all_users(model, all_users_df, all_items_df, user_history, meta_data, device, top_k=200, batch_size=128):
    """
    Generates recalls for all users for a single model source.
    This is an adapted version of the `evaluate_dssm` logic.
    """
    model.eval()
    
    user_cols = meta_data['user_feature_cols']
    item_cols = meta_data['item_feature_cols']
    iid_col = meta_data['item_feature_cols'][0]

    # 1. Get all item embeddings
    all_item_inputs = torch.tensor(all_items_df[item_cols].values, dtype=torch.long).to(device)
    item_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_item_inputs), batch_size * 4), desc="Calculating item embeddings"):
            batch_items = all_item_inputs[i:i + batch_size * 4]
            vec = model.get_item_vector(batch_items)
            item_vecs.append(vec)
    all_item_vecs = torch.cat(item_vecs, dim=0)
    
    # Map item ID to its index in the all_items_df for quick lookup
    iid_to_idx = {iid: idx for idx, iid in enumerate(all_items_df[iid_col])}
    
    user_recalls = {}
    
    # 2. Iterate through all users in batches
    user_inputs = torch.tensor(all_users_df[user_cols].values, dtype=torch.long)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(user_inputs), batch_size), desc="Generating user recalls"):
            batch_user_inputs = user_inputs[i:i+batch_size].to(device)
            batch_user_ids = all_users_df.iloc[i:i+batch_size][meta_data['user_feature_cols'][0]].tolist()
            
            # Get user embeddings for the batch
            user_vecs = model.get_user_vector(batch_user_inputs)
            
            # Calculate scores against all items
            scores = torch.matmul(user_vecs, all_item_vecs.t())
            
            # Apply filtering for each user in the batch
            for j, uid in enumerate(batch_user_ids):
                history_items = user_history.get(uid, set())
                if history_items:
                    history_indices = [iid_to_idx[iid] for iid in history_items if iid in iid_to_idx]
                    if history_indices:
                        scores[j, history_indices] = -float('inf')
            
            # Get top K for the batch
            _, top_indices_batch = torch.topk(scores, top_k)
            
            # Store results
            top_indices_batch_cpu = top_indices_batch.cpu().numpy()
            for j, uid in enumerate(batch_user_ids):
                recommended_iids = all_items_df.iloc[top_indices_batch_cpu[j]][iid_col].tolist()
                user_recalls[uid] = recommended_iids
                
    return user_recalls

def generate_and_save_hot_list(redis_client, data_config, top_k=500):
    """Calculates and saves the global hot list based on global_score."""
    root_dir = get_project_root()
    processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])
    global_score_path = os.path.join(processed_dir, data_config['global_score_file'])
    
    if not os.path.exists(global_score_path):
        print(f"Error: Global score file not found at {global_score_path}")
        return
        
    print("Loading global scores...")
    global_score_df = pd.read_csv(global_score_path)
    
    # The file should contain 'iid' and 'global_score'
    hot_list_df = global_score_df.sort_values(by='global_score', ascending=False).head(top_k)
    item_scores = dict(zip(hot_list_df['iid'], hot_list_df['global_score']))
    
    redis_client.save_global_hot_list(item_scores)
    print(f"Saved {len(item_scores)} items to the global hot list.")


def main(args):
    """Main orchestration function."""
    
    root_dir = get_project_root()
    data_config = load_config(os.path.join(root_dir, 'config/data.yaml'))
    
    meta_data_path = os.path.join(root_dir, data_config['processed_data_dir'], data_config['meta_data_file'])
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    redis_client = RedisStorage(db=args.redis_db)
    print(f"Connected to Redis DB: {args.redis_db}")

    # --- Generate and Save User Recalls ---
    if args.recall_sources:
        print("Loading all data to build user/item profiles...")
        full_df, all_users_df, all_items_df = load_all_data(data_config, meta_data)
        user_history = build_user_history(full_df, meta_data)
        
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {device}")
        
        for source in args.recall_sources:
            print(f"--- Generating recalls for source: {source} ---")
            
            model_config_path = os.path.join(root_dir, f'config/{source}.yaml')
            model_config = load_config(model_config_path)

            print("Loading model...")
            model = DSSM(meta_data['feature_max_idx'], **model_config['model_params'])
            model_path = os.path.join(root_dir, 'models/saved', f"{model_config['model_name']}_{model_config['training_method']}_best.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            user_recalls = generate_recalls_for_all_users(
                model, all_users_df, all_items_df, user_history, meta_data, device, top_k=args.recall_top_k
            )
            
            print(f"Generated recalls for {len(user_recalls)} users. Saving to Redis...")
            for user_id, item_ids in tqdm(user_recalls.items(), desc=f"Saving {source} recalls"):
               redis_client.save_user_recall_results(user_id, item_ids, recall_source=source)
            
            print(f"Finished processing source: {source}")
            
    # --- Generate and Save Global Hot List ---
    if args.gen_hot_list:
        print("\n--- Generating and saving global hot list ---")
        generate_and_save_hot_list(redis_client, data_config, top_k=args.hot_list_top_k)
        print("Finished saving global hot list.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline recall generation and saving jobs.")
    parser.add_argument(
        '--recall-sources', 
        nargs='+', 
        choices=['dssm_pointwise', 'dssm_inbatch'], 
        help='A list of recall sources to generate.'
    )
    parser.add_argument(
        '--gen-hot-list', 
        action='store_true', 
        help='If set, generates and saves the global hot list.'
    )
    parser.add_argument('--redis-db', type=int, default=5, help='Redis database to use.')
    parser.add_argument('--hot-list-top-k', type=int, default=500, help='Number of items in the global hot list.')
    parser.add_argument('--recall-top-k', type=int, default=200, help='Number of items per user recall list.')

    args = parser.parse_args()

    if not args.recall_sources and not args.gen_hot_list:
        print("Error: Must specify at least one job to run. Use --recall-sources or --gen-hot-list.")
        parser.print_help()
    else:
        main(args)
