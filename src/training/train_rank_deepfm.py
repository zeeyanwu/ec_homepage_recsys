import sys
import os
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score
from src.models.ranking.deepfm import DeepFM

from root import get_root_dir
os.chdir(get_root_dir())

# Configuration
DATA_DIR = 'data/processed'
MODEL_DIR = 'src/models/saved'
BATCH_SIZE = 512
EPOCHS = 5
LR = 0.001

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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
            # slot_id -> global_score
            self.global_score_map = dict(zip(score_df['slot_id'].astype(int), score_df['global_score']))
        
        # Identify item id column
        self.iid_col = 'iid'
        
        # Prepare Tensors
        self.sparse_data = torch.tensor(self.data[self.all_sparse_cols].values, dtype=torch.long)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.float32)
        
        # Prepare Dense Features
        self.dense_data = torch.zeros((len(self.data), 1), dtype=torch.float32)
        
        if self.global_score_map and self.iid_col in self.data.columns:
            # Map iid (slot_id) to global_score
            # fillna with 0 for unseen items
            scores = self.data[self.iid_col].map(self.global_score_map).fillna(0.0).values
            self.dense_data = torch.tensor(scores.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.sparse_data[idx], self.dense_data[idx], self.labels[idx]

def train(train_loader, test_loader, total_vocab_size, num_sparse_features):
    print("\n=== Training Ranking Model (DeepFM) ===")
    model = DeepFM(
        total_vocab_size=total_vocab_size,
        num_sparse_features=num_sparse_features,
        num_dense_features=1, # global_score
        embedding_dim=16,
        hidden_dims=[64, 32],
        dropout=0.3
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for sparse_inputs, dense_inputs, labels in train_loader:
            optimizer.zero_grad()
            preds = model(sparse_inputs, dense_inputs)
            loss = criterion(preds.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # For AUC
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            
        train_auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train AUC: {train_auc:.4f}")
        
        # Evaluation
        evaluate(model, test_loader)
        
    return model

def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for sparse_inputs, dense_inputs, labels in test_loader:
            preds = model(sparse_inputs, dense_inputs)
            loss = criterion(preds.squeeze(), labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    auc = roc_auc_score(all_labels, all_preds)
    print(f"   >>> Test Loss: {total_loss/len(test_loader):.4f} | Test AUC: {auc:.4f}")

def main():
    # 1. Load Metadata
    meta_path = os.path.join(DATA_DIR, 'meta_data.pkl')
    if not os.path.exists(meta_path):
        print("Meta data not found.")
        return
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        
    total_vocab_size = meta['feature_dims']
    user_cols = meta['user_feature_cols']
    item_cols = meta['item_feature_cols']
    
    # 2. Datasets
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    global_score_csv = os.path.join(DATA_DIR, 'item_global_score.csv')
    
    train_dataset = RankDataset(train_csv, user_cols, item_cols, global_score_csv)
    test_dataset = RankDataset(test_csv, user_cols, item_cols, global_score_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Train
    model = train(train_loader, test_loader, total_vocab_size, len(user_cols) + len(item_cols))
    
    # 4. Save
    save_path = os.path.join(MODEL_DIR, 'deepfm.pth')
    torch.save(model.state_dict(), save_path)
    print(f"DeepFM Model saved to {save_path}")

if __name__ == "__main__":
    main()
