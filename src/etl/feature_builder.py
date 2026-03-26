import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

def build_feature_map(df, user_feature_cols, item_feature_cols):
    """
    Builds a global feature map (slot-based) for all categorical features.
    The keys will have prefixes like 'user_uid' or 'item_iid'.
    """
    print("\n[2] Building Global Feature Map...")
    
    feature_map = {}
    idx = 1  # Start from 1, 0 is reserved for padding/unknown

    # Combine all feature columns
    all_feature_cols = user_feature_cols + item_feature_cols

    for col in tqdm(all_feature_cols, desc="Building feature map"):
        unique_vals = df[col.split('_')[-1]].unique() # Assumes column name in df is without prefix
        for val in unique_vals:
            key = f"{col}={val}"
            if key not in feature_map:
                feature_map[key] = idx
                idx += 1
    
    print(f"Total Unique Features (Slots): {idx}")
    return feature_map, idx

def transform_data_with_feature_map(df, feature_map, user_feature_cols, item_feature_cols):
    """
    Transforms the dataframe by mapping feature values to their slot IDs.
    """
    print("\n[3] Transforming Data with Feature Map...")
    df_transformed = pd.DataFrame()
    
    # Combine all feature columns
    all_feature_cols = user_feature_cols + item_feature_cols

    for col in tqdm(all_feature_cols, desc="Mapping features to IDs"):
        df_col_name = col.split('_')[-1] # Assumes column name in df is without prefix
        df_transformed[col] = df[df_col_name].apply(lambda x: feature_map.get(f"{col}={x}", 0))

    # Keep label and timestamp if they exist
    if 'label' in df.columns:
        df_transformed['label'] = df['label']
    if 'ts' in df.columns:
        df_transformed['ts'] = df['ts']
        
    return df_transformed

def compute_and_save_global_score(original_df, train_indices, config, root_dir, feature_map):
    """
    Computes and saves the global item hotness score.
    """
    print("\n[5] Computing Global Item Hotness Score...")
    processed_dir = os.path.join(root_dir, config['processed_data_dir'])
    
    # Use only training data to compute scores to avoid leakage
    train_df = original_df.loc[train_indices]

    item_impressions = train_df['iid'].value_counts().reset_index()
    item_impressions.columns = ['iid', 'impression_count']
    
    item_clicks = train_df[train_df['label'] == 1]['iid'].value_counts().reset_index()
    item_clicks.columns = ['iid', 'click_count']

    df_score = pd.merge(item_impressions, item_clicks, on='iid', how='left').fillna(0)
    
    global_ctr = df_score['click_count'].sum() / df_score['impression_count'].sum()
    C = 100
    df_score['ctr_mean'] = (df_score['click_count'] + C * global_ctr) / (df_score['impression_count'] + C)

    df_score['slot_id'] = df_score['iid'].apply(lambda x: feature_map.get(f"item_iid={x}", 0)) # Assumes item iid has prefix
    
    output_path = os.path.join(processed_dir, config['global_score_file'])
    df_score.to_csv(output_path, columns=['iid', 'slot_id', 'ctr_mean', 'global_score'], index=False)
    print(f"Global scores saved to {output_path}")
    return df_score
