import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
# from root import get_root_dir
# os.chdir(get_root_dir())
import sys
from tqdm import tqdm
import pickle

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.recall.dssm import DSSM

# Hyperparameters
BATCH_SIZE = 1024
EPOCHS = 20
LR = 0.001
NEG_RATIO = 5 # 1 Positive : 5 Negatives (Undersampling)

DATA_DIR = 'data/processed'
MODEL_DIR = 'src/models/saved'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class RecallDataset(Dataset):
    def __init__(self, data_path, user_cols, item_cols, neg_ratio=5, is_train=True):
        self.data = pd.read_csv(data_path)
        self.user_cols = user_cols
        self.item_cols = item_cols
        
        if is_train:
            # Separate Positives and Negatives
            if 'label' in self.data.columns:
                pos_df = self.data[self.data['label'] > 0].copy()
                pos_df['label'] = 1.0
                
                neg_df = self.data[self.data['label'] == 0].copy()
                neg_df['label'] = 0.0
                
                # Subsample Negatives
                n_pos = len(pos_df)
                if len(neg_df) > n_pos * neg_ratio:
                    print(f"Subsampling Negatives: {len(neg_df)} -> {n_pos * neg_ratio}")
                    neg_df = neg_df.sample(n=n_pos * neg_ratio, random_state=42)
                
                print(f"Dataset Stats: Pos={len(pos_df)}, Neg={len(neg_df)} (Ratio 1:{len(neg_df)/len(pos_df):.1f})")
                self.data = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            else:
                print("Warning: No label column found in train data.")
        else:
            # For Test, we only keep Positives for Recall Evaluation
            if 'label' in self.data.columns:
                self.data = self.data[self.data['label'] > 0].reset_index(drop=True)
                
        self.user_data = torch.tensor(self.data[user_cols].values, dtype=torch.long)
        self.item_data = torch.tensor(self.data[item_cols].values, dtype=torch.long)
        if 'label' in self.data.columns:
            self.labels = torch.tensor(self.data['label'].values, dtype=torch.float)
        else:
            self.labels = torch.zeros(len(self.data), dtype=torch.float)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.user_data[idx], self.item_data[idx], self.labels[idx]

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def efficient_evaluate(model, test_loader, train_df, item_cols, k_list=[10, 50, 100, 200, 500]):
    print("\n=== Evaluating Recall ===")
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Get all unique items and their embeddings
    # Deduplicate items from train_df to get unique item features
    item_feats = train_df[item_cols].drop_duplicates(subset=[item_cols[0]])
    all_item_inputs = torch.tensor(item_feats.values, dtype=torch.long).to(device)
    
    # Compute item vectors in batches
    item_vecs = []
    batch = 1024
    with torch.no_grad():
        for i in range(0, len(all_item_inputs), batch):
            batch_items = all_item_inputs[i:i+batch]
            vec = model.get_item_vector(batch_items)
            item_vecs.append(vec)
    all_item_vecs = torch.cat(item_vecs, dim=0) # (NumItems, EmbedDim)
    
    # Map Slot ID to Matrix Index for fast lookup
    # all_item_inputs[:, 0] is the IID Slot ID
    iid_slot_to_idx = {slot.item(): idx for idx, slot in enumerate(all_item_inputs[:, 0])}
    
    # 2. Build User History for filtering
    uid_col_name = train_df.columns[1] # Typically 'uid'
    iid_col_name = item_cols[0]        # Typically 'iid' column name
    
    # Filter only positive history
    if 'label' in train_df.columns:
        pos_history = train_df[train_df['label']>0]
    else:
        pos_history = train_df # Fallback
        
    user_history = pos_history.groupby(uid_col_name)[iid_col_name].apply(set).to_dict()
    
    hits = {k: 0 for k in k_list}
    total_users = 0
    
    with torch.no_grad():
        # Unpack 3 values: user, item, label
        for user_inputs, target_item_inputs, _ in test_loader:
            batch_size = user_inputs.size(0)
            user_vecs = model.get_user_vector(user_inputs.to(device))
            
            # (Batch, NumItems)
            scores = torch.matmul(user_vecs, all_item_vecs.t())
            
            # For each user
            for i in range(batch_size):
                u_id_slot = user_inputs[i][0].item()
                target_iid_slot = target_item_inputs[i][0].item()
                
                # Mask history items
                history_items = user_history.get(u_id_slot, set())
                history_indices = [iid_slot_to_idx[s] for s in history_items if s in iid_slot_to_idx]
                
                user_scores = scores[i].clone()
                user_scores[history_indices] = -float('inf')
                
                # Restore target item score if masked
                if target_iid_slot in iid_slot_to_idx:
                    target_idx = iid_slot_to_idx[target_iid_slot]
                    user_scores[target_idx] = scores[i][target_idx]
                else:
                    continue # Target not in candidates
                
                # Top K
                max_k = max(k_list)
                _, top_indices = torch.topk(user_scores, max_k)
                top_indices = top_indices.tolist()
                
                # Check hit
                if target_idx in top_indices:
                    rank = top_indices.index(target_idx)
                    for k in k_list:
                        if rank < k:
                            hits[k] += 1
                
                total_users += 1
                
    recall_50 = 0.0
    for k in k_list:
        if total_users > 0:
            recall = hits[k]/total_users
            print(f"Recall@{k}: {recall:.4f}")
            if k == 50:
                recall_50 = recall
    return recall_50

def train(train_loader, test_loader, train_df, item_cols, total_vocab_size, user_cols_count, item_cols_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    model = DSSM(
        total_vocab_size=total_vocab_size, 
        embedding_dim=64, 
        hidden_dims=[256, 128],
        user_feature_count=user_cols_count, 
        item_feature_count=item_cols_count
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Handle Class Imbalance
    pos_weight = torch.tensor([NEG_RATIO], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    print("\n=== Training Recall Model (DSSM) with Pointwise Loss ===")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0
        step = 0
        for user_inputs, item_inputs, labels in train_loader:
            user_inputs = user_inputs.to(device)
            item_inputs = item_inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, user_vectors, item_vectors = model(user_inputs, item_inputs)
            
            # Pointwise Score (Cosine Similarity * Temperature)
            # DSSM vectors are normalized, so dot product is cosine
            scores = torch.sum(user_vectors * item_vectors, dim=1)
            scores = scores * 10.0 # Temperature scaling
            
            loss = criterion(scores, labels)
            
            # Acc
            preds = (scores > 0).float() # Sigmoid(0) = 0.5
            acc = (preds == labels).sum().item() / labels.size(0)
            total_acc += acc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/step:.4f}, Batch Acc: {total_acc/step:.4f}")
        
        recall_50 = efficient_evaluate(model, test_loader, train_df, item_cols)
        
        early_stopping(recall_50)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return model

def main():
    # Load metadata
    with open(os.path.join(DATA_DIR, 'meta_data.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    feature_dims = meta['feature_dims']
    user_cols = meta['user_feature_cols']
    item_cols = meta['item_feature_cols']
    
    # Load Data
    print("Loading Data...")
    train_dataset = RecallDataset(os.path.join(DATA_DIR, 'train.csv'), user_cols, item_cols, neg_ratio=NEG_RATIO, is_train=True)
    test_dataset = RecallDataset(os.path.join(DATA_DIR, 'test.csv'), user_cols, item_cols, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load full train df for evaluation filtering
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    
    model = train(train_loader, test_loader, train_df, item_cols, feature_dims, len(user_cols), len(item_cols))
    
    save_path = os.path.join(MODEL_DIR, 'dssm_pointwise.pth')
    torch.save(model.state_dict(), save_path)
    print(f"DSSM Pointwise Model saved to {save_path}")

if __name__ == '__main__':
    main()
