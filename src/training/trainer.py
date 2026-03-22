
import torch
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Assuming these custom dataset classes are defined elsewhere and are compatible
# We will need to create these or adapt existing ones.
from src.training.datasets import RankDataset, RecallDataset
from src.utils.config_loader import get_project_root


def train_model(model, config, data_config, meta_data):
    """
    A generic training function for DeepFM and DSSM models.

    Args:
        model (torch.nn.Module): The model instance (DeepFM or DSSM).
        config (dict): The model-specific configuration.
        data_config (dict): The data configuration.
        meta_data (dict): The metadata dictionary.

    Returns:
        tuple: (best_metric, model_save_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model.to(device)
    print(f"Using device: {device}")

    # --- Data Loading ---
    root_dir = get_project_root()
    processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])
    train_path = os.path.join(processed_dir, data_config['train_file'])
    test_path = os.path.join(processed_dir, data_config['test_file'])

    if config['model_name'] == 'deepfm':
        global_score_path = os.path.join(processed_dir, data_config['global_score_file'])
        train_dataset = RankDataset(train_path, meta_data['user_feature_cols'], meta_data['item_feature_cols'], global_score_path)
        test_dataset = RankDataset(test_path, meta_data['user_feature_cols'], meta_data['item_feature_cols'], global_score_path)
    elif config['model_name'] == 'dssm':
        train_dataset = RecallDataset(
            train_path, 
            meta_data['user_feature_cols'], 
            meta_data['item_feature_cols'],
            training_method=config['training_method'],
            neg_ratio=config.get('neg_ratio', 5), # .get() for safety, not present in in-batch
            is_train=True
        )
        # For DSSM evaluation, we need the full training data to filter seen items
        full_train_df = pd.read_csv(train_path)
        full_test_df = pd.read_csv(test_path)
        test_dataset = RecallDataset(test_path, meta_data['user_feature_cols'], meta_data['item_feature_cols'], is_train=False)
    else:
        raise NotImplementedError

    pin_memory_flag = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=pin_memory_flag)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=pin_memory_flag)

    # --- Optimizer and Criterion ---
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if config['model_name'] == 'deepfm':
        criterion = torch.nn.BCELoss()
    elif config['model_name'] == 'dssm':
        if config['training_method'] == 'pointwise':
            pos_weight = torch.tensor([config['neg_ratio']], dtype=torch.float, device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif config['training_method'] == 'in_batch':
            criterion = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    best_metric = 0.0
    best_epoch = 0
    
    model_dir = os.path.join(root_dir, 'models/saved')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{config['model_name']}_{config['training_method']}_best.pth") # Add method to name

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch in progress_bar:
            optimizer.zero_grad()
            
            if config['model_name'] == 'deepfm':
                sparse_inputs, dense_inputs, labels = [b.to(device) for b in batch]
                preds = model(sparse_inputs, dense_inputs)
                loss = criterion(preds.squeeze(), labels)
            elif config['model_name'] == 'dssm':
                user_inputs, item_inputs, labels = [b.to(device) for b in batch]
                
                if config['training_method'] == 'pointwise':
                    _, user_vecs, item_vecs = model(user_inputs, item_inputs)
                    scores = torch.sum(user_vecs * item_vecs, dim=1)
                    loss = criterion(scores, labels)
                elif config['training_method'] == 'in_batch':
                    _, user_vecs, item_vecs = model(user_inputs, item_inputs)
                    # In-batch loss calculation
                    scores = torch.matmul(user_vecs, item_vecs.t())
                    # Target is the diagonal
                    targets = torch.arange(user_inputs.size(0)).to(device)
                    loss = criterion(scores, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # --- Evaluation ---
        if config['model_name'] == 'deepfm':
            test_auc = evaluate_deepfm(model, test_loader, device)
            print(f"   >>> Test AUC: {test_auc:.4f}")
            current_metric = test_auc
        elif config['model_name'] == 'dssm':
            recall_50 = evaluate_dssm(model, test_loader, full_train_df, full_test_df, meta_data, device)
            current_metric = recall_50 # Using Recall@50 as the key metric

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"   >>> New Best Model Saved (Metric: {best_metric:.4f}) at Epoch {best_epoch}")

    print(f"\nTraining finished. Best metric: {best_metric:.4f}")
    return best_metric, model_save_path


def evaluate_deepfm(model, test_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for sparse_inputs, dense_inputs, labels in test_loader:
            sparse_inputs, dense_inputs, labels = sparse_inputs.to(device), dense_inputs.to(device), labels.to(device)
            preds = model(sparse_inputs, dense_inputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return roc_auc_score(all_labels, all_preds)


def evaluate_dssm(model, test_loader, train_df, test_df, meta_data, device, k_list=[50, 100]):
    model.eval()
    item_cols = meta_data['item_feature_cols']
    
    # --- FIX: Build candidate pool from ALL items in train and test sets ---
    all_items_df = pd.concat([train_df[item_cols], test_df[item_cols]]).drop_duplicates()
    all_item_inputs = torch.tensor(all_items_df.values, dtype=torch.long).to(device)
    # ---------------------------------------------------------------------

    item_vecs = []
    with torch.no_grad():
        # Process in batches to avoid OOM on large item sets
        for i in range(0, len(all_item_inputs), 1024):
            batch_items = all_item_inputs[i:i+1024]
            vec = model.get_item_vector(batch_items)
            item_vecs.append(vec)
    all_item_vecs = torch.cat(item_vecs, dim=0)
    
    # Map the unique item ID (slot) to its index in the embedding tensor
    iid_slot_to_idx = {slot.item(): idx for idx, slot in enumerate(all_item_inputs[:, 0])}

    # 2. Build User History of seen items from the training set
    uid_col_name = train_df.columns[1] # Assumes user id is the second column
    iid_col_name = item_cols[0] # Assumes item id is the first item feature
    user_history = train_df[train_df['label']>0].groupby(uid_col_name)[iid_col_name].apply(set).to_dict()

    hits = {k: 0 for k in k_list}
    total_users = 0

    with torch.no_grad():
        for user_inputs, target_item_inputs, _ in test_loader:
            user_vecs = model.get_user_vector(user_inputs.to(device))
            scores = torch.matmul(user_vecs, all_item_vecs.t())
            
            for i in range(user_inputs.size(0)):
                u_id_slot = user_inputs[i][0].item()
                target_iid_slot = target_item_inputs[i][0].item()
                history_items = user_history.get(u_id_slot, set())
                history_indices = [iid_slot_to_idx[s] for s in history_items if s in iid_slot_to_idx]
                
                user_scores = scores[i]
                user_scores[history_indices] = -float('inf')
                
                if target_iid_slot not in iid_slot_to_idx: continue
                target_idx = iid_slot_to_idx[target_iid_slot]

                _, top_indices = torch.topk(user_scores, max(k_list))
                top_indices = top_indices.tolist()

                if target_idx in top_indices:
                    rank = top_indices.index(target_idx)
                    for k in k_list:
                        if rank < k: hits[k] += 1
                total_users += 1
    
    for k in sorted(hits.keys()):
        recall = hits[k] / total_users if total_users > 0 else 0
        print(f"   >>> Recall@{k}: {recall:.4f}")
        
    return hits.get(50, 0) / total_users if total_users > 0 else 0

