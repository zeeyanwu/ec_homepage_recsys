import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.recall.dssm import DSSM
from src.serving.redis_storage import RedisStorage

# from root import get_root_dir
# os.chdir(get_root_dir())

DATA_DIR = 'data/processed'
MODEL_DIR = 'src/models/saved'
BATCH_SIZE = 1024

def load_model(meta_path, model_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    total_vocab_size = meta['feature_dims']
    user_cols_count = len(meta['user_feature_cols'])
    item_cols_count = len(meta['item_feature_cols'])
    
    model = DSSM(
        total_vocab_size=total_vocab_size,
        embedding_dim=64,
        hidden_dims=[256, 128],
        user_feature_count=user_cols_count,
        item_feature_count=item_cols_count
    )
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model.eval()
    return model, meta

def get_item_embeddings(model, df, item_cols):
    """
    Generate embeddings for all unique items
    """
    print("Generating Item Embeddings...")
    unique_item_df = df[item_cols].drop_duplicates().reset_index(drop=True)
    item_tensor = torch.tensor(unique_item_df.values, dtype=torch.long)
    
    embeddings = []
    # Batch processing
    with torch.no_grad():
        for i in range(0, len(item_tensor), BATCH_SIZE):
            batch = item_tensor[i:i+BATCH_SIZE]
            emb = model.get_item_vector(batch)
            embeddings.append(emb)
            
    all_embeddings = torch.cat(embeddings, dim=0)
    
    # We need to map row index back to Item ID (slot id of the iid column)
    # Assuming 'iid' is part of the item_cols and we want to return a dict {iid: vector}
    # However, 'iid' in the dataframe is already a slot index.
    # We will return the unique_item_df (with iids) and the tensor
    return unique_item_df, all_embeddings

def export_recall_results(redis_client, model, meta, train_df, test_df, top_k=50):
    """
    Compute and export Recall results for all users
    """
    user_cols = meta['user_feature_cols']
    item_cols = meta['item_feature_cols']
    
    # 1. Prepare Candidate Items (from Train data)
    candidate_df, candidate_vectors = get_item_embeddings(model, train_df, item_cols)
    print(f"Candidate Items: {len(candidate_df)}")
    
    # Extract actual IIDs (assuming first column of item_cols is IID or we can find it)
    # In our preprocessing, all cols are slot indices. 
    # We need to know which column represents the Item ID to store in Redis.
    # Usually the first column in item_cols is the Item ID slot.
    # Let's verify with a simple heuristic or use the whole tuple as ID if needed.
    # For now, we use the value in the first item column as the Item ID (Slot ID).
    candidate_ids = candidate_df.iloc[:, 0].values.tolist()
    
    # 2. Identify Target Users (Unique users from Train + Test)
    # We only need their features.
    all_data = pd.concat([train_df, test_df], axis=0)
    unique_users = all_data[user_cols].drop_duplicates().reset_index(drop=True)
    print(f"Target Users: {len(unique_users)}")
    
    user_tensor = torch.tensor(unique_users.values, dtype=torch.long)
    
    # 3. Compute Scores & Save
    print(f"Computing Top-{top_k} for each user and saving to Redis...")
    
    # Process users in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(user_tensor), BATCH_SIZE)):
            user_batch = user_tensor[i:i+BATCH_SIZE]
            
            # (Batch, Dim)
            user_vecs = model.get_user_vector(user_batch)
            
            # (Batch, Num_Candidates)
            scores = torch.matmul(user_vecs, candidate_vectors.t())
            
            # Top-K
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
            
            # Convert to lists
            top_indices = top_indices.cpu().numpy()
            top_scores = top_scores.cpu().numpy()
            
            # Save to Redis
            for j in range(len(user_batch)):
                # Get User ID (first col of user features)
                u_row = user_batch[j].cpu().numpy()
                uid = str(u_row[0]) # Use Slot ID as User ID
                
                # Get Rec Items
                rec_indices = top_indices[j]
                rec_scores = top_scores[j]
                
                rec_iids = [str(candidate_ids[idx]) for idx in rec_indices]
                
                # Save
                redis_client.save_recall_results(
                    user_id=uid, 
                    item_ids=rec_iids, 
                    scores=rec_scores,
                    prefix='dssm:recall:'
                )

def export_merged_recall(redis_client, model, meta, train_df, test_df, slot_to_raw_item, top_k_dssm=250, hot_top_n=100):
    """
    Export merged recall list for each user:
    1) DSSM Top-K candidates per user
    2) Merged with Global Hot Top-N (deduplicated)
    Results are stored in Redis under key: recall:{uid}
    
    Args:
        slot_to_raw_item (dict): Mapping from Item Slot ID (int) to Raw Item ID (str)
    """
    user_cols = meta['user_feature_cols']
    item_cols = meta['item_feature_cols']
    
    # 1. Prepare Candidate Items
    candidate_df, candidate_vectors = get_item_embeddings(model, train_df, item_cols)
    print(f"Candidate Items: {len(candidate_df)}")
    candidate_ids = candidate_df.iloc[:, 0].astype(int).values.tolist()
    
    # 2. Load Global Hot Top-N
    hot_csv_path = os.path.join(DATA_DIR, 'item_global_score.csv')
    hot_item_ids = []
    if os.path.exists(hot_csv_path):
        hot_df = pd.read_csv(hot_csv_path)
        if 'slot_id' in hot_df.columns and 'global_score' in hot_df.columns:
            hot_df = hot_df.sort_values('global_score', ascending=False).head(hot_top_n)
            # Store as Raw IDs directly
            # hot_item_ids = hot_df['slot_id'].astype(int).tolist()
            # Map to Raw
            hot_slot_ids = hot_df['slot_id'].astype(int).tolist()
            hot_item_ids = [slot_to_raw_item.get(sid, str(sid)) for sid in hot_slot_ids]
            
            print(f"Loaded {len(hot_item_ids)} global hot items (Raw IDs) from {hot_csv_path}")
        else:
            print("Warning: item_global_score.csv missing 'slot_id' or 'global_score' columns. Hot items disabled.")
    else:
        print(f"Warning: {hot_csv_path} not found. Hot items disabled.")
    
    # 3. Identify Target Users
    all_data = pd.concat([train_df, test_df], axis=0)
    unique_users = all_data[user_cols].drop_duplicates().reset_index(drop=True)
    print(f"Target Users: {len(unique_users)}")
    user_tensor = torch.tensor(unique_users.values, dtype=torch.long)
    
    # 4. Compute DSSM Scores and Merge with Hot
    print(f"Computing DSSM Top-{top_k_dssm} and merging with Hot Top-{hot_top_n} for each user...")
    with torch.no_grad():
        for i in tqdm(range(0, len(user_tensor), BATCH_SIZE)):
            user_batch = user_tensor[i:i+BATCH_SIZE]
            user_vecs = model.get_user_vector(user_batch)
            scores = torch.matmul(user_vecs, candidate_vectors.t())
            
            top_scores, top_indices = torch.topk(scores, k=top_k_dssm, dim=1)
            top_indices = top_indices.cpu().numpy()
            
            for j in range(len(user_batch)):
                u_row = user_batch[j].cpu().numpy()
                uid_slot = int(u_row[0])
                # TODO: If we had User Raw Map, we would convert here.
                # For now, we use Slot ID as User Key.
                # If we want to simulate Raw ID, we can prefix it or just keep it.
                # Given user request "Store using Raw UID", we strictly need the map.
                # Since we lack the map, I will use "u{slot}" as a placeholder for Raw ID 
                # OR keep it as is if that's what we have.
                # Let's keep it as is (Slot ID) for User Key because we can't invent Raw IDs.
                # But for Item IDs, we MUST convert.
                
                uid = str(uid_slot)
                
                rec_indices = top_indices[j]
                
                # Interleave Strategy: 4 DSSM + 1 Hot
                # DSSM: rec_indices -> Slot IDs -> Raw IDs
                
                dssm_slot_ids = [candidate_ids[int(idx)] for idx in rec_indices]
                dssm_items = [slot_to_raw_item.get(sid, str(sid)) for sid in dssm_slot_ids]
                
                # Hot items are already Raw IDs
                hot_items = hot_item_ids 
                
                merged_items = []
                seen = set()
                
                # Pointers
                p_dssm = 0
                p_hot = 0
                
                while len(merged_items) < (top_k_dssm + hot_top_n) and (p_dssm < len(dssm_items) or p_hot < len(hot_items)):
                    # Add 4 DSSM items
                    for _ in range(4):
                        if p_dssm < len(dssm_items):
                            item = dssm_items[p_dssm]
                            if item not in seen:
                                seen.add(item)
                                merged_items.append(item)
                            p_dssm += 1
                    
                    # Add 1 Hot item
                    if p_hot < len(hot_items):
                        item = hot_items[p_hot]
                        if item not in seen:
                            seen.add(item)
                            merged_items.append(item)
                        p_hot += 1
                
                # Assign descending scores to preserve order in Redis ZSET
                n = len(merged_items)
                merged_scores = list(range(n, 0, -1))
                
                redis_client.save_recall_results(
                    user_id=uid,
                    item_ids=merged_items,
                    scores=merged_scores,
                    prefix='recall:'
                )

def export_global_hot(redis_client, hot_csv_path, slot_to_raw_item, top_n=100):
    print("Exporting Global Hot Items...")
    if not os.path.exists(hot_csv_path):
        print(f"Warning: {hot_csv_path} not found. Skipping.")
        return
        
    df = pd.read_csv(hot_csv_path)
    
    if 'slot_id' not in df.columns or 'global_score' not in df.columns:
        print("Error: item_global_score.csv missing required columns.")
        return
        
    # Sort by global_score desc
    df = df.sort_values('global_score', ascending=False).head(top_n)
    
    # Map Slot ID -> Raw ID
    slot_ids = df['slot_id'].astype(int).tolist()
    items = [slot_to_raw_item.get(sid, str(sid)) for sid in slot_ids]
    
    scores = df['global_score'].tolist()
    
    redis_client.save_recall_results(
        user_id='global_hot', 
        item_ids=items, 
        scores=scores, 
        prefix=''
    )
    print(f"Saved {len(items)} hot items (Raw IDs) to Redis key 'global_hot'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=6379)
    args = parser.parse_args()
    
    # 0. Init Redis
    try:
        r = RedisStorage(host=args.host, port=args.port, db=1)
        r.client.ping()
        print("Connected to Redis.")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        print("Please ensure Redis is running.")
        return

    # 1. Load Resources
    meta_path = os.path.join(DATA_DIR, 'meta_data.pkl')
    model_path = os.path.join(MODEL_DIR, 'dssm_pointwise.pth')
    
    model, meta = load_model(meta_path, model_path)
    
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Load Raw -> Slot Mapping for Items
    item_map_df = pd.read_csv(os.path.join(DATA_DIR, 'item_global_score.csv'))
    slot_to_raw_item = dict(zip(item_map_df['slot_id'].astype(int), item_map_df['iid'].astype(str)))

    # 2. Export merged Recall (DSSM + Global Hot)
    export_merged_recall(r, model, meta, train_df, test_df, slot_to_raw_item, top_k_dssm=250, hot_top_n=100)
    
    # 3. Export Global Hot for cold-start fallback
    hot_csv = os.path.join(DATA_DIR, 'item_global_score.csv')
    export_global_hot(r, hot_csv, slot_to_raw_item)
    
    print("\n=== Export Completed ===")

if __name__ == '__main__':
    main()
