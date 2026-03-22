
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class RankDataset(Dataset):
    def __init__(self, data_path, user_cols, item_cols, global_score_path=None):
        print(f"Loading data from {data_path} for Ranking...")
        self.data = pd.read_csv(data_path)
        
        self.user_cols = user_cols
        self.item_cols = item_cols
        self.all_sparse_cols = user_cols + item_cols
        
        # Load Global Scores
        self.global_score_map = {}
        if global_score_path and os.path.exists(global_score_path):
            print(f"Loading global scores from {global_score_path}...")
            score_df = pd.read_csv(global_score_path)
            self.global_score_map = dict(zip(score_df['slot_id'].astype(int), score_df['global_score']))
        
        self.iid_col = 'iid'
        
        # Prepare Tensors
        self.sparse_data = torch.tensor(self.data[self.all_sparse_cols].values, dtype=torch.long)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.float32)
        
        # Prepare Dense Features
        self.dense_data = torch.zeros((len(self.data), 1), dtype=torch.float32)
        
        if self.global_score_map and self.iid_col in self.data.columns:
            scores = self.data[self.iid_col].map(self.global_score_map).fillna(0.0).values
            self.dense_data = torch.tensor(scores.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.sparse_data[idx], self.dense_data[idx], self.labels[idx]


class RecallDataset(Dataset):
    def __init__(self, data_path, user_cols, item_cols, training_method='pointwise', neg_ratio=5, is_train=True):
        self.data = pd.read_csv(data_path)
        self.user_cols = user_cols
        self.item_cols = item_cols
        
        if is_train:
            # For training, we need to handle negative sampling based on the method
            if 'label' in self.data.columns:
                pos_df = self.data[self.data['label'] > 0].copy()
                
                if training_method == 'pointwise':
                    pos_df['label'] = 1.0
                    neg_df = self.data[self.data['label'] == 0].copy()
                    neg_df['label'] = 0.0
                    
                    n_pos = len(pos_df)
                    if len(neg_df) > n_pos * neg_ratio:
                        neg_df = neg_df.sample(n=n_pos * neg_ratio, random_state=42)
                    
                    print(f"Pointwise Dataset: Pos={len(pos_df)}, Neg={len(neg_df)} (Ratio 1:{len(neg_df)/len(pos_df):.1f})")
                    self.data = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                
                elif training_method == 'in_batch':
                    print(f"In-Batch Dataset: Using {len(pos_df)} positive pairs.")
                    self.data = pos_df.reset_index(drop=True)
        else:
            # For evaluation, we only care about positive interactions
            if 'label' in self.data.columns:
                self.data = self.data[self.data['label'] > 0].reset_index(drop=True)
                
        self.user_data = torch.tensor(self.data[user_cols].values, dtype=torch.long)
        self.item_data = torch.tensor(self.data[item_cols].values, dtype=torch.long)
        if 'label' in self.data.columns and training_method == 'pointwise':
            self.labels = torch.tensor(self.data['label'].values, dtype=torch.float)
        else:
            # For in-batch, the label is implicitly the index, so we don't need to return it.
            # For evaluation, we also don't need a specific label.
            self.labels = torch.zeros(len(self.data), dtype=torch.float)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.user_data[idx], self.item_data[idx], self.labels[idx]
