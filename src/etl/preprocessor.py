import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from root import get_root_dir
os.chdir(get_root_dir())

class RecSysPreprocessor:
    def __init__(self, data_dir='data/raw_data', processed_dir='data/processed'):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
        self.feature_map = {} # Global Slot: "col_name=value" -> int_id
        self.feature_dims = 0
        
        # Define feature columns structure
        self.user_feature_cols = ['uid', 'utag1', 'utag2']
        self.item_feature_cols = ['iid', 'itag1', 'itag2', 'itag3']
        self.all_feature_cols = self.user_feature_cols + self.item_feature_cols
        
    def run(self):
        """执行完整的 ETL 流程"""
        print("=== ETL Started ===")
        
        # 1. Load & Merge
        df = self.load_and_merge()
        
        # 2. Build Feature Map (Global Slot)
        self.build_feature_map(df)
        
        # 3. Transform Data (Map to IDs)
        df_transformed = self.transform_data(df)
        
        # 4. Split Train/Test (Time-based per user)
        train_df, test_df = self.split_and_save(df_transformed)
        
        # 5. Compute Global Score (using ONLY Train data to avoid leakage)
        
        self.compute_global_score(df, train_df.index)
        
        print("=== ETL Finished ===")

    def load_and_merge(self):
        print("[1] Loading and Merging Data...")
        
        # Load Shop Log
        shop_path = os.path.join(self.data_dir, 'shop.dat')
        df_shop = pd.read_csv(shop_path, header=None, names=['ts', 'uid', 'iid', 'label'])
        
        # Load User Features
        user_path = os.path.join(self.data_dir, 'user_feature.dat')
        df_user = pd.read_csv(user_path, header=None, names=['uid', 'utag1', 'utag2'])
        
        # Load Item Features
        item_path = os.path.join(self.data_dir, 'item_feature.dat')
        df_item = pd.read_csv(item_path, header=None, names=['iid', 'itag1', 'itag2', 'itag3'])
        
        # Merge
        # Left join to keep all interactions
        df = df_shop.merge(df_user, on='uid', how='left')
        df = df.merge(df_item, on='iid', how='left')
        
        # Fill NaN (if any) with a special token, though EDA showed 100% coverage
        df.fillna(0, inplace=True) 
        
        # Reorder columns: ts, user_features..., item_features..., label
        cols = ['ts'] + self.user_feature_cols + self.item_feature_cols + ['label']
        df = df[cols]
        
        print(f"Merged Data Shape: {df.shape}")
        return df

    def build_feature_map(self, df):
        print("[2] Building Global Feature Map...")
        
        # Iterate over all feature columns to build the global dictionary
        idx = 0
        for col in self.all_feature_cols:
            unique_vals = df[col].unique()
            for val in unique_vals:
                key = f"{col}={val}"
                if key not in self.feature_map:
                    self.feature_map[key] = idx
                    idx += 1
        
        self.feature_dims = idx
        print(f"Total Unique Features (Slots): {self.feature_dims}")
        
        # Save feature map
        with open(os.path.join(self.processed_dir, 'feature_map.pkl'), 'wb') as f:
            pickle.dump(self.feature_map, f)

    def transform_data(self, df):
        print("[3] Transforming Data to IDs...")
        
        df_transformed = df.copy()
        
        # Apply mapping
        # Optimized approach: use map() for each column
        for col in self.all_feature_cols:
            # Create a temporary series of keys
            keys = df[col].apply(lambda x: f"{col}={x}")
            # Map keys to IDs
            df_transformed[col] = keys.map(self.feature_map)
            
        return df_transformed

    def split_and_save(self, df):
        print("[4] Splitting Train/Test (8:2 by time per user)...")
        
        # Sort by user and time
        df = df.sort_values(['uid', 'ts'])
        
        # Calculate rank percentage per user
        # method='first' ensures unique ranks for identical timestamps
        df['rank'] = df.groupby('uid')['ts'].rank(method='first', pct=True)
        
        train_mask = df['rank'] <= 0.8
        test_mask = ~train_mask
        
        train_df = df[train_mask].drop(columns=['rank'])
        test_df = df[test_mask].drop(columns=['rank'])
        
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Save
        train_df.to_csv(os.path.join(self.processed_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, 'test.csv'), index=False)
        
        # Save feature dimensions
        with open(os.path.join(self.processed_dir, 'meta_data.pkl'), 'wb') as f:
            meta = {
                'feature_dims': self.feature_dims,
                'user_feature_cols': self.user_feature_cols,
                'item_feature_cols': self.item_feature_cols
            }
            pickle.dump(meta, f)
            
        return train_df, test_df

    def compute_global_score(self, original_df, train_indices, alpha=0.5):
        print("[5] Computing Global Item Scores (Hotness)...")
        
        # Let's be safer: Recalculate the split on original_df using the same logic
        # OR better: The `original_df` passed to this method is the one BEFORE transformation
        # We should apply the same sorting and splitting logic.
        
        df = original_df.copy()
        df = df.sort_values(['uid', 'ts'])
        df['rank'] = df.groupby('uid')['ts'].rank(method='first', pct=True)
        train_df = df[df['rank'] <= 0.8].copy()
        max_ts = train_df['ts'].max()
        min_ts = train_df['ts'].min()
        time_range = max_ts - min_ts + 1e-8
        train_df['age'] = (max_ts - train_df['ts']) / time_range
        decay_factor = 3.0
        train_df['time_weight'] = np.exp(-decay_factor * train_df['age'])
        train_df['label_weighted'] = train_df['label'] * train_df['time_weight']
        item_stats = train_df.groupby('iid').agg(
            impression_count=('time_weight', 'sum'),
            label_weighted_sum=('label_weighted', 'sum')
        ).reset_index()
        
        # Bayesian Smoothing for CTR
        # Global CTR = Total Weighted Labels / Total Weighted Impressions
        global_ctr = train_df['label_weighted'].sum() / (train_df['time_weight'].sum() + 1e-8)
        
        # Smoothing factor
        m = 20.0
        item_stats['smoothed_ctr'] = (item_stats['label_weighted_sum'] + m * global_ctr) / (item_stats['impression_count'] + m)
        
        # Normalize to [0, 1] for fusion
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-8)
            
        item_stats['norm_count'] = normalize(item_stats['impression_count'])
        item_stats['norm_ctr'] = normalize(item_stats['smoothed_ctr'])
        
        # Global Score = alpha * Popularity + (1-alpha) * CTR
        item_stats['global_score'] = (
            alpha * item_stats['norm_count'] + 
            (1 - alpha) * item_stats['norm_ctr']
        )
        
        # Keep raw stats for debugging
        item_stats['ctr_mean'] = item_stats['smoothed_ctr']
        
        # Map original IID to Slot ID
        # self.feature_map keys are like "iid=123"
        def get_slot_id(original_iid):
            key = f"iid={original_iid}"
            return self.feature_map.get(key, -1) # -1 if not found (shouldn't happen for train set)
            
        item_stats['slot_id'] = item_stats['iid'].apply(get_slot_id)
        
        # Save
        output_path = os.path.join(self.processed_dir, 'item_global_score.csv')
        # Rename impression_count back to something standard or keep it?
        # Let's keep impression_count in the CSV for clarity.
        item_stats[['iid', 'slot_id', 'global_score', 'impression_count', 'ctr_mean']].to_csv(output_path, index=False)
        print(f"Global scores saved to {output_path}")
        print(item_stats[['iid', 'slot_id', 'global_score']].head())

if __name__ == "__main__":
    preprocessor = RecSysPreprocessor()
    preprocessor.run()
