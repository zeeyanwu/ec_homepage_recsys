import pandas as pd
import os

DATA_DIR = 'data/raw_data'

def run_eda():
    print("=== Starting EDA ===")
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    shop_df = pd.read_csv(os.path.join(DATA_DIR, 'shop.dat'), header=None, names=['ts', 'uid', 'iid', 'label'])
    item_df = pd.read_csv(os.path.join(DATA_DIR, 'item_feature.dat'), header=None, names=['iid', 'itag1', 'itag2', 'itag3'])
    user_df = pd.read_csv(os.path.join(DATA_DIR, 'user_feature.dat'), header=None, names=['uid', 'utag1', 'utag2'])
    
    print(f"Shop Log Shape: {shop_df.shape}")
    print(f"Item Feature Shape: {item_df.shape}")
    print(f"User Feature Shape: {user_df.shape}")
    
    # 2. Label Distribution
    print("\n[2] Label Distribution (CTR):")
    label_counts = shop_df['label'].value_counts()
    ctr = shop_df['label'].mean()
    print(label_counts)
    print(f"Overall CTR: {ctr:.4f}")
    
    # 3. Cardinality
    print("\n[3] Cardinality Analysis:")
    n_users_log = shop_df['uid'].nunique()
    n_items_log = shop_df['iid'].nunique()
    n_users_feat = user_df['uid'].nunique()
    n_items_feat = item_df['iid'].nunique()
    
    print(f"Unique Users in Log: {n_users_log}")
    print(f"Unique Items in Log: {n_items_log}")
    print(f"Unique Users in Feature Table: {n_users_feat}")
    print(f"Unique Items in Feature Table: {n_items_feat}")
    
    # 4. Feature Coverage (Match Rate)
    print("\n[4] Feature Coverage:")
    # Check how many log users exist in user_feature
    valid_users = shop_df[shop_df['uid'].isin(user_df['uid'])]
    user_coverage = len(valid_users) / len(shop_df)
    print(f"User Feature Coverage (Log-level): {user_coverage:.2%}")
    
    # Check how many log items exist in item_feature
    valid_items = shop_df[shop_df['iid'].isin(item_df['iid'])]
    item_coverage = len(valid_items) / len(shop_df)
    print(f"Item Feature Coverage (Log-level): {item_coverage:.2%}")
    
    # 5. Missing Values
    print("\n[5] Missing Values Check:")
    print("User Features Missing:")
    print(user_df.isnull().sum())
    print("Item Features Missing:")
    print(item_df.isnull().sum())
    
    # 6. Sample Data
    print("\n[6] Data Samples:")
    print("--- Shop Log ---")
    print(shop_df.head(3))
    print("--- User Feature ---")
    print(user_df.head(3))
    print("--- Item Feature ---")
    print(item_df.head(3))

if __name__ == "__main__":
    run_eda()
