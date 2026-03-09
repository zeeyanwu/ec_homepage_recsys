import pandas as pd
import os
from tqdm import tqdm

def split_and_save_data(df_transformed, config, root_dir):
    """
    Splits the data into training and test sets based on user history and saves them.
    """
    print("\n[4] Splitting and Saving Train/Test Data...")
    processed_dir = os.path.join(root_dir, config['processed_data_dir'])

    # Sort by user and timestamp to get user's historical behavior
    df_transformed.sort_values(by=['uid', 'ts'], inplace=True)

    # Get the last interaction for each user for the test set
    test_indices = df_transformed.groupby('uid').tail(1).index
    train_indices = df_transformed.index.difference(test_indices)

    train_df = df_transformed.loc[train_indices]
    test_df = df_transformed.loc[test_indices]

    # Save files
    train_path = os.path.join(processed_dir, config['train_file'])
    test_path = os.path.join(processed_dir, config['test_file'])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to {train_path} ({len(train_df)} rows)")
    print(f"Test data saved to {test_path} ({len(test_df)} rows)")
    
    return train_indices, test_indices
