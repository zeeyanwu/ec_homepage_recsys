import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def split_and_save_data(df_transformed, config, root_dir):
    """
    Splits the data into training and test sets based on user history and saves them.
    """
    print("\n[4] Splitting and Saving Train/Test Data...")
    processed_dir = os.path.join(root_dir, config['processed_data_dir'])

    # --- User-Level 80/20 Chronological Split ---
    # For each user, the earliest 80% of interactions are for training, 
    # and the latest 20% are for testing. This ensures a strict temporal
    # separation and prevents data leakage, while making sure users in the
    # test set also have a history in the training set.

    # Ensure data is sorted by user and time
    df_transformed.sort_values(by=['uid', 'ts'], inplace=True)
    
    train_dfs = []
    test_dfs = []

    # Group by user and perform the split for each
    # tqdm adds a progress bar for better visibility
    grouped = df_transformed.groupby('uid')

    for _, group in tqdm(grouped, desc="Splitting user data"):
        n_interactions = len(group)
        
        # If a user has only one interaction, it must go into the training set.
        if n_interactions < 2:
            train_dfs.append(group)
            continue
        
        # We want approximately 20% for the test set, but always at least 1.
        # np.floor ensures we don't round up to take too many test samples.
        n_test_samples = max(1, int(np.floor(n_interactions * 0.2)))
        
        # The first (n - n_test) samples are for training
        train_dfs.append(group.head(n_interactions - n_test_samples))
        
        # The last n_test samples are for testing
        test_dfs.append(group.tail(n_test_samples))

    # Concatenate the lists of dataframes into the final train and test sets
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    # Get the indices for returning, though they are not strictly needed anymore
    train_indices = train_df.index
    test_indices = test_df.index

    # Save files
    train_path = os.path.join(processed_dir, config['train_file'])
    test_path = os.path.join(processed_dir, config['test_file'])

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to {train_path} ({len(train_df)} rows)")
    print(f"Test data saved to {test_path} ({len(test_df)} rows)")
    
    return train_indices, test_indices
