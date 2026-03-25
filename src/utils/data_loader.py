
import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils.config_loader import get_project_root

class RecSysDataManager:
    """
    Manages all data loading, processing, and DataLoader creation for the recommendation system.
    """
    def __init__(self, data_config):
        self.config = data_config
        self.root_dir = get_project_root()
        self.processed_data_dir = os.path.join(self.root_dir, self.config['processed_data_dir'])
        self.meta_data = None
        self.feature_map = None

    def _load_meta(self):
        """Loads metadata and feature map from pickled files."""
        with open(os.path.join(self.processed_data_dir, self.config['meta_file']), 'rb') as f:
            self.meta_data = pickle.load(f)
        with open(os.path.join(self.processed_data_dir, self.config['feature_map_file']), 'rb') as f:
            self.feature_map = pickle.load(f)
        
        # Create a full item pool for evaluation
        self.item_pool = torch.arange(self.feature_map['iid']).long()

    def prepare_dataloaders(self):
        """
        Loads train/test data and prepares PyTorch DataLoaders.
        Returns:
            tuple: (train_loader, val_loader, test_loader, feature_map, item_pool)
        """
        self._load_meta()

        # Load datasets
        train_df = pd.read_csv(os.path.join(self.processed_data_dir, self.config['train_file']))
        test_df = pd.read_csv(os.path.join(self.processed_data_dir, self.config['test_file']))

        # Convert to PyTorch tensors
        train_tensors = self._df_to_tensors(train_df)
        test_tensors = self._df_to_tensors(test_df)

        train_dataset = TensorDataset(*train_tensors)
        test_dataset = TensorDataset(*test_tensors)

        # Split train data for validation
        train_size = int(len(train_dataset) * (1 - self.config['val_split_ratio']))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config.get('num_workers', 0))
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config.get('num_workers', 0))
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config.get('num_workers', 0))

        print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
        
        return train_loader, val_loader, test_loader, self.feature_map, self.item_pool

    def _df_to_tensors(self, df):
        """Converts a pandas DataFrame to a list of PyTorch tensors."""
        user_features = ['uid', 'utag1', 'utag2']
        item_features = ['iid', 'itag1', 'itag2', 'itag3']
        
        user_tensor = torch.tensor(df[user_features].values, dtype=torch.long)
        item_tensor = torch.tensor(df[item_features].values, dtype=torch.long)
        labels = torch.tensor(df['label'].values, dtype=torch.float32)
        
        return [user_tensor, item_tensor, labels]
