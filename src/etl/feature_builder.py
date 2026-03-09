import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

def build_feature_map(df, user_feature_cols, item_feature_cols):
    """
    Builds a global feature map (slot-based) for all categorical features.
    """
    print("\n[2] Building Global Feature Map...")
    feature_map = {}
    idx = 0
    all_feature_cols = user_feature_cols + item_feature_cols

    for col in all_feature_cols:
        unique_vals = df[col].unique()
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
    all_feature_cols = user_feature_cols + item_feature_cols

    for col in tqdm(all_feature_cols, desc="Mapping features to IDs"):
        key_prefix = f"{col}="
        df_transformed[col] = df[col].apply(lambda x: feature_map.get(key_prefix + str(x)))

    df_transformed['label'] = df['label']
    df_transformed['ts'] = df['ts']
    return df_transformed

def compute_and_save_global_score(original_df, train_indices, config, root_dir, feature_map, alpha=0.5):
    """
    Computes and saves the global item hotness score.
    """
    print("\n[5] Computing Global Item Hotness Score...")
    processed_dir = os.path.join(root_dir, config['processed_data_dir'])

    # Use only training data to compute scores to avoid leakage
    train_df = original_df.loc[train_indices]

    # Impression and Click counts
    item_impressions = train_df['iid'].value_counts().reset_index()
    item_impressions.columns = ['iid', 'impression_count']
    
    item_clicks = train_df[train_df['label'] == 1]['iid'].value_counts().reset_index()
    item_clicks.columns = ['iid', 'click_count']

    # Merge and calculate CTR
    df_score = pd.merge(item_impressions, item_clicks, on='iid', how='left').fillna(0)
    
    # Smoothed CTR (Bayesian smoothing)
    global_ctr = df_score['click_count'].sum() / df_score['impression_count'].sum()
    C = 100 # Confidence parameter, a common choice
    df_score['ctr_mean'] = (df_score['click_count'] + C * global_ctr) / (df_score['impression_count'] + C)

    # Time Decay Factor
    max_ts = train_df['ts'].max()
    item_last_ts = train_df.groupby('iid')['ts'].max().reset_index()
    item_last_ts['time_decay'] = np.exp(-alpha * (max_ts - item_last_ts['ts']) / (24 * 3600))
    df_score = pd.merge(df_score, item_last_ts[['iid', 'time_decay']], on='iid', how='left')

    # Global Score
    df_score['global_score'] = np.log10(df_score['impression_count']) * df_score['ctr_mean'] * df_score['time_decay']
    df_score['global_score'] = (df_score['global_score'] - df_score['global_score'].min()) / \
                              (df_score['global_score'].max() - df_score['global_score'].min())

    # Add slot_id for serving lookup
    df_score['slot_id'] = df_score['iid'].apply(lambda x: feature_map.get(f'iid={x}'))
    
    # Save to file
    output_path = os.path.join(processed_dir, config['global_score_file'])
    df_score.to_csv(output_path, index=False)
    print(f"Global scores saved to {output_path}")
    return df_score
