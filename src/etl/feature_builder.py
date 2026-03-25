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

    # --- Exponential Freshness Score ---
    # This score represents the "freshness" or "recency" of an item. The higher the score, the more recent the item.
    # It's calculated using an exponential decay function based on the item's last interaction time.
    
    # Map the last interaction time for each item to the score dataframe
    last_ts_map = train_df.groupby('iid')['ts'].max()
    df_score['last_interaction_time'] = df_score['iid'].map(last_ts_map)
    df_score['last_interaction_time'] = df_score['last_interaction_time'].fillna(0) # Items not in train_df get 0

    reference_time = df_score['last_interaction_time'].max() # The latest interaction time is our "now"
    
    # Define a half-life for the score, e.g., 7 days.
    # After 7 days, the item's time score is reduced to 50%.
    half_life_days = 7.0
    half_life_seconds = half_life_days * 24 * 3600
    
    # Convert timestamps from ms to seconds and calculate age
    age_in_seconds = (reference_time - df_score['last_interaction_time']) / 1000
    
    # Calculate decay rate (lambda) from half-life
    # The formula is: T_half = ln(2) / lambda  =>  lambda = ln(2) / T_half
    decay_rate = np.log(2) / half_life_seconds
    
    df_score['freshness_score'] = np.exp(-decay_rate * age_in_seconds)
    
    # Ensure items that never appeared (last_interaction_time was 0) get a freshness_score of 0
    df_score.loc[df_score['last_interaction_time'] == 0, 'freshness_score'] = 0
    # Fill any remaining NaNs from items not in training data with 0 decay
    df_score['freshness_score'] = df_score['freshness_score'].fillna(0)

    # Final global_score: A robust combination of quality (ctr_mean) and freshness (freshness_score)
    # The additive model is more robust than a multiplicative one, as it prevents a single low score
    # (e.g., a low freshness_score for an old but high-quality item) from zeroing out the entire score.
    df_score['global_score'] = 0.5 * df_score['ctr_mean'] + 0.5 * df_score['freshness_score']

    # Add slot_id for serving lookup
    df_score['slot_id'] = df_score['iid'].apply(lambda x: feature_map.get(f'iid={x}'))
    
    # Save to file
    output_path = os.path.join(processed_dir, config['global_score_file'])
    df_score.to_csv(output_path, columns=['iid', 'slot_id', 'ctr_mean', 'freshness_score', 'global_score'], index=False)
    print(f"Global scores saved to {output_path}")
    return df_score
