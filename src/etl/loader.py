import pandas as pd
import os

def load_and_merge_data(config, root_dir):
    """
    Loads shop logs, user features, and item features, then merges them.
    """
    print("[1] Loading and Merging Data...")

    raw_data_dir = os.path.join(root_dir, config['raw_data_dir'])

    # Load Shop Log
    shop_path = os.path.join(raw_data_dir, config['shop_log_file'])
    df_shop = pd.read_csv(shop_path, header=None, names=['ts', 'uid', 'iid', 'label'])

    # Load User Features
    user_path = os.path.join(raw_data_dir, config['user_feature_file'])
    df_user = pd.read_csv(user_path, header=None, names=['uid', 'utag1', 'utag2'])

    # Load Item Features
    item_path = os.path.join(raw_data_dir, config['item_feature_file'])
    df_item = pd.read_csv(item_path, header=None, names=['iid', 'itag1', 'itag2', 'itag3'])

    # Merge dataframes
    df = pd.merge(df_shop, df_user, on='uid', how='left')
    df = pd.merge(df, df_item, on='iid', how='left')

    # Fill missing feature values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)

    print(f"Data loaded and merged. Total rows: {len(df)}")
    return df
