import sys
import os
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from root import get_root_dir
os.chdir(get_root_dir())
import pickle
import os
import torch.nn.functional as F
from src.models.recall.dssm import DSSM

# Configuration
DATA_DIR = 'data/processed'
MODEL_DIR = 'src/models/saved'
BATCH_SIZE = 256 # Smaller batch size for better generalization
EPOCHS = 20
LR = 0.001

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class RecallDataset(Dataset):
    def __init__(self, data_path, user_cols, item_cols):
        print(f"Loading data from {data_path} for Recall...")
        df = pd.read_csv(data_path)
        
        # Filter only positive samples for Recall training
        # We rely on in-batch negatives or random negatives
        self.data = df[df['label'] > 0.0].reset_index(drop=True)
        print(f"Filtered positive samples: {len(self.data)}")
        
        self.user_cols = user_cols
        self.item_cols = item_cols
        
        # Create 0-based indices for Matrix Masking
        self.le_user = LabelEncoder()
        self.le_item = LabelEncoder()
        # Use the first column of user_cols/item_cols as the ID column
        self.data['user_idx'] = self.le_user.fit_transform(self.data[user_cols[0]]) 
        self.data['item_idx'] = self.le_item.fit_transform(self.data[item_cols[0]])
        
        self.num_users = len(self.le_user.classes_)
        self.num_items = len(self.le_item.classes_)
        
        # Build Dense Interaction Matrix
        self.interaction_matrix = torch.zeros((self.num_users, self.num_items), dtype=torch.bool)
        u_indices = torch.tensor(self.data['user_idx'].values, dtype=torch.long)
        i_indices = torch.tensor(self.data['item_idx'].values, dtype=torch.long)
        self.interaction_matrix[u_indices, i_indices] = True
        
        self.user_data = torch.tensor(self.data[user_cols].values, dtype=torch.long)
        self.item_data = torch.tensor(self.data[item_cols].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.user_data[idx], self.item_data[idx], self.data.iloc[idx]['user_idx'], self.data.iloc[idx]['item_idx']

def efficient_evaluate(model, test_loader, train_df, item_cols, k_list=[10, 50]):
    print("\n=== Evaluating Recall ===")
    model.eval()
    
    # 1. Build Candidate Pool
    unique_item_df = train_df[item_cols].drop_duplicates().reset_index(drop=True)
    # Create a map: tuple(features) -> pool_index
    item_to_idx = {tuple(row): i for i, row in enumerate(unique_item_df.values)}
    
    item_tensor = torch.tensor(unique_item_df.values, dtype=torch.long)
    
    item_vectors = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(item_tensor), batch_size):
            batch = item_tensor[i:i+batch_size]
            vecs = model.get_item_vector(batch)
            item_vectors.append(vecs)
    candidate_vectors = torch.cat(item_vectors, dim=0) #(M, Dim)
    
    # 2. Evaluate
    hits = {k: 0 for k in k_list}
    total = 0
    
    with torch.no_grad():
        for user_inputs, target_item_inputs, _, _ in test_loader:
            batch_size = user_inputs.size(0)
            user_vecs = model.get_user_vector(user_inputs)
            
            # Similarity: (Batch, M)
            scores = torch.matmul(user_vecs, candidate_vectors.t())
            
            # Find rank of ground truth
            # First, find ground truth index in pool
            target_np = target_item_inputs.cpu().numpy()
            ground_truth_indices = []
            valid_mask = [] # Some test items might not be in training set (Cold Start)
            
            for row in target_np:
                idx = item_to_idx.get(tuple(row), -1)
                ground_truth_indices.append(idx)
                valid_mask.append(idx != -1)
                
            ground_truth_indices = torch.tensor(ground_truth_indices, device=scores.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=scores.device)
            
            # Filter valid samples
            if not valid_mask.any():
                continue
                
            valid_scores = scores[valid_mask]
            valid_gt_indices = ground_truth_indices[valid_mask]
            
            # Get Top-Max(K)
            max_k = max(k_list)
            _, topk_indices = torch.topk(valid_scores, k=max_k, dim=1) # (ValidBatch, MaxK)
            
            for k in k_list:
                # Check if gt index is in top k columns
                # topk_indices[:, :k] shape (ValidBatch, k)
                # valid_gt_indices shape (ValidBatch)
                # We need to broadcast compare
                hit_matrix = (topk_indices[:, :k] == valid_gt_indices.unsqueeze(1))
                hits[k] += hit_matrix.any(dim=1).sum().item()
                
            total += valid_mask.sum().item()
            
    metrics = {}
    for k in k_list:
        recall = hits[k] / total if total > 0 else 0.0
        metrics[k] = recall
        print(f"Recall@{k}: {recall:.4f}")
        
    return metrics

def train(train_loader, test_loader, train_df, item_cols, total_vocab_size, user_cols_count, item_cols_count):
    print("\n=== Training Recall Model (DSSM) with In-batch Negatives ===")
    model = DSSM(
        total_vocab_size=total_vocab_size, 
        embedding_dim=64, 
        hidden_dims=[256, 128],
        user_feature_count=user_cols_count, 
        item_feature_count=item_cols_count
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # CrossEntropyLoss expects logits (scores) and class indices
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    interaction_matrix = None
    for epoch in range(EPOCHS):
        total_loss = 0
        step = 0
        for user_inputs, item_inputs, batch_u_idx, batch_i_idx in train_loader:
            optimizer.zero_grad()
            
            if interaction_matrix is None:
                 interaction_matrix = train_loader.dataset.interaction_matrix.to(user_inputs.device)

            # Forward pass: get vectors
            # dssm forward returns (score, u_vec, i_vec)
            # We need vectors to compute full similarity matrix
            _, user_vectors, item_vectors = model(user_inputs, item_inputs)
            
            # Compute Similarity Matrix: (Batch, Batch)
            # Row i contains scores of user i against all items in batch
            scores = torch.matmul(user_vectors, item_vectors.t())
            
            # Scale scores (temperature)
            scores = scores * 5.0 
            
            # Masking: Identify false negatives in batch
            mask = interaction_matrix[batch_u_idx][:, batch_i_idx]
            # Keep diagonal (Self) as Positive
            mask.fill_diagonal_(False)
            # Apply Mask
            scores.masked_fill_(mask, -1e9)
            
            # Targets: The diagonal elements are the positives (0, 1, 2, ..., B-1)
            targets = torch.arange(scores.size(0)).to(scores.device)
            
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/step:.4f}")
        
        # Evaluate at the end of each epoch
        efficient_evaluate(model, test_loader, train_df, item_cols)
        model.train()
        
    return model



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
    
    # 2. Dataset
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_csv = os.path.join(DATA_DIR, 'test.csv')
    
    # Load raw dfs for candidate pool building
    train_df = pd.read_csv(train_csv)
    
    train_dataset = RecallDataset(train_csv, user_cols, item_cols)
    test_dataset = RecallDataset(test_csv, user_cols, item_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Train
    model = train(train_loader, test_loader, train_df, item_cols, total_vocab_size, len(user_cols), len(item_cols))
    
    # 4. Evaluate (Final)
    efficient_evaluate(model, test_loader, train_df, item_cols)
    
    # 5. Save
    save_path = os.path.join(MODEL_DIR, 'dssm_inbatch.pth')
    torch.save(model.state_dict(), save_path)
    print(f"DSSM Model saved to {save_path}")

if __name__ == "__main__":
    main()
