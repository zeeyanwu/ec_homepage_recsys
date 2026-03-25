
import torch
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import mlflow

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
        # For DSSM, we need full train/test dataframes for evaluation
        full_train_df = pd.read_csv(train_path)
        full_test_df = pd.read_csv(test_path)

        train_dataset = RecallDataset(
            full_train_df,
            meta_data['user_feature_cols'],
            meta_data['item_feature_cols'],
            training_method=config['training_method'],
            neg_ratio=config.get('neg_ratio', 0),
            is_train=True
        )
        test_dataset = RecallDataset(
            full_test_df,
            meta_data['user_feature_cols'],
            meta_data['item_feature_cols'],
            training_method=config['training_method'],
            is_train=False
        )
        
        # --- For random negative sampling in in-batch mode ---
        all_item_features_tensor = None
        if config['training_method'] == 'in_batch' and config.get('random_negative_count', 0) > 0:
            print("Preparing for random negative sampling...")
            item_cols = meta_data['item_feature_cols']
            iid_col = meta_data['item_feature_cols'][0] 
            all_items_df = pd.concat([full_train_df[item_cols], full_test_df[item_cols]]).drop_duplicates(subset=[iid_col])
            all_item_features_tensor = torch.tensor(all_items_df.values, dtype=torch.long)
            all_item_ids_tensor = torch.tensor(all_items_df[iid_col].values, dtype=torch.long) # FIX: Create iid tensor
            print(f"Created a pool of {len(all_item_features_tensor)} unique items for random negative sampling.")

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

    # Prepare for masking known positives if enabled
    user_history = None
    if config.get("training_method") == "in_batch" and config.get("mask_known_positives", False):
        print("Preparing for masking known positives...")
        user_history = full_train_df[full_train_df['label'] > 0].groupby('uid')['iid'].apply(set).to_dict()
        print(f"Created user history for {len(user_history)} users.")

    # --- Training Loop ---
    best_metric = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = config.get('early_stopping_patience', 5) # Default patience is 5 epochs
    
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
                # Unpack all data from the loader
                if config['training_method'] == 'in_batch':
                    user_inputs, item_inputs, _, batch_uids, batch_iids = batch
                    user_inputs, item_inputs = user_inputs.to(device), item_inputs.to(device)
                else: # pointwise
                    user_inputs, item_inputs, labels = [b.to(device) for b in batch]

                if config['training_method'] == 'pointwise':
                    _, user_vecs, item_vecs = model(user_inputs, item_inputs)
                    scores = torch.sum(user_vecs * item_vecs, dim=1)
                    loss = criterion(scores, labels)
                
                elif config['training_method'] == 'in_batch':
                    _, user_vecs, positive_item_vecs = model(user_inputs, item_inputs)
                    
                    candidate_item_vecs = positive_item_vecs
                    candidate_iids = batch_iids

                    random_negative_count = config.get('random_negative_count', 0)
                    if random_negative_count > 0 and all_item_features_tensor is not None:
                        rand_indices = torch.randint(0, len(all_item_ids_tensor), (random_negative_count,))
                        random_negative_features = all_item_features_tensor[rand_indices].to(device)
                        random_negative_iids = all_item_ids_tensor[rand_indices]
                        
                        random_negative_vecs = model.get_item_vector(random_negative_features)
                        
                        candidate_item_vecs = torch.cat([positive_item_vecs, random_negative_vecs], dim=0)
                        candidate_iids = torch.cat([batch_iids, random_negative_iids.to(batch_iids.device)])

                    scores = torch.matmul(user_vecs, candidate_item_vecs.t())

                    # --- Mask known positives ---
                    if user_history is not None:
                        # Reverted to the simple, effective loop-based masking that yielded the best results.
                        batch_size = user_inputs.size(0)
                        uids_list = batch_uids.cpu().tolist()
                        iids_list = batch_iids.cpu().tolist()

                        for i, uid in enumerate(uids_list):
                            if uid in user_history:
                                known_positive_iids = user_history[uid]
                                # In-batch negatives
                                for j, iid in enumerate(iids_list):
                                    # Mask if it's a known positive AND not the true positive for this row
                                    if iid in known_positive_iids and i != j:
                                        scores[i, j] = -1e9
                                # Random negatives
                                random_negative_count = config.get('random_negative_count', 0)
                                if random_negative_count > 0 and all_item_ids_tensor is not None:
                                    # The `rand_indices` tensor was created before this block
                                    for k, rand_idx in enumerate(rand_indices):
                                        rand_iid = all_item_ids_tensor[rand_idx].item()
                                        if rand_iid in known_positive_iids:
                                            scores[i, batch_size + k] = -1e9

                    targets = torch.arange(user_inputs.size(0)).to(device)
                    loss = criterion(scores, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # --- Evaluation ---
        metric_name = ''
        if config['model_name'] == 'deepfm':
            test_auc, test_logloss = evaluate_deepfm(model, test_loader, device)
            print(f"   >>> Test AUC: {test_auc:.4f}")
            print(f"   >>> Test LogLoss: {test_logloss:.4f}")
            current_metric = test_auc
            metric_name = 'auc'
            mlflow.log_metric("epoch_log_loss", test_logloss, step=epoch+1)

        elif config['model_name'] == 'dssm':
            recall_50, recall_100 = evaluate_dssm(model, test_loader, full_train_df, full_test_df, meta_data, device)
            current_metric = recall_50 # Using Recall@50 as the key metric
            metric_name = 'recall_at_50'
            mlflow.log_metric("epoch_recall_at_100", recall_100, step=epoch+1)

        # --- Log metrics to MLflow for this epoch ---
        if metric_name:
            mlflow.log_metric(f"epoch_avg_loss", avg_loss, step=epoch+1)
            mlflow.log_metric(f"epoch_{metric_name}", current_metric, step=epoch+1)

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_no_improve = 0 # Reset counter
            torch.save(model.state_dict(), model_save_path)
            print(f"   >>> New Best Model Saved (Metric: {best_metric:.4f}) at Epoch {best_epoch}")
        else:
            epochs_no_improve += 1

        # --- Early Stopping Check ---
        if epochs_no_improve >= patience:
            print(f"\n   !!! Early stopping triggered after {patience} epochs with no improvement.")
            break

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
    test_auc = roc_auc_score(all_labels, all_preds)
    test_logloss = log_loss(all_labels, all_preds)
    return test_auc, test_logloss


def evaluate_dssm(model, test_loader, train_df, test_df, meta_data, device, k_list=[50, 100]):
    model.eval()
    item_cols = meta_data['item_feature_cols']
    uid_col = meta_data['user_feature_cols'][0]  # Assuming first user feature is uid
    iid_col = meta_data['item_feature_cols'][0]  # Assuming first item feature is iid

    # --- FIX 1: Build candidate pool from ALL items in train and test sets ---
    all_items_df = pd.concat([train_df[item_cols], test_df[item_cols]]).drop_duplicates(subset=[iid_col]).reset_index(drop=True)
    all_item_inputs = torch.tensor(all_items_df.values, dtype=torch.long).to(device)

    item_vecs = []
    with torch.no_grad():
        # Process in batches to avoid OOM on large item sets
        for i in range(0, len(all_item_inputs), 1024):
            batch_items = all_item_inputs[i:i+1024]
            vec = model.get_item_vector(batch_items)
            item_vecs.append(vec)
    all_item_vecs = torch.cat(item_vecs, dim=0)
    
    # Map the unique item ID to its index in the embedding tensor
    iid_to_idx = {iid.item(): idx for idx, iid in enumerate(all_item_inputs[:, 0])}

    # 2. Build User History of seen items from the training set
    user_history = train_df[train_df['label'] > 0].groupby(uid_col)[iid_col].apply(set).to_dict()

    # --- FIX 2: Denominator for Recall should be the number of positive items in the test set ---
    true_positives_per_user = test_df[test_df['label'] > 0].groupby(uid_col)[iid_col].apply(set).to_dict()
    total_positive_items = sum(len(v) for v in true_positives_per_user.values())

    hits = {k: 0 for k in k_list}
    
    # Group test loader by user
    test_user_batches = {}
    for user_inputs, item_inputs, labels in test_loader:
        for i in range(len(user_inputs)):
            uid = user_inputs[i, 0].item()
            if uid not in test_user_batches:
                test_user_batches[uid] = {'user_inputs': [], 'item_inputs': [], 'labels': []}
            test_user_batches[uid]['user_inputs'].append(user_inputs[i])
            test_user_batches[uid]['item_inputs'].append(item_inputs[i])
            test_user_batches[uid]['labels'].append(labels[i])

    print("   Evaluating Recall...")
    with torch.no_grad():
        for uid, batch in test_user_batches.items():
            # We only need one user vector
            user_input_tensor = torch.stack(batch['user_inputs'])[0:1].to(device)
            user_vec = model.get_user_vector(user_input_tensor)

            # Score against all items
            scores = torch.matmul(user_vec, all_item_vecs.t()).squeeze()
            
            # Remove seen items
            history_items = user_history.get(uid, set())
            history_indices = [iid_to_idx[iid] for iid in history_items if iid in iid_to_idx]
            scores[history_indices] = -float('inf')

            # Get top K recommendations
            _, top_indices = torch.topk(scores, max(k_list))
            
            # Get the set of recommended item IDs
            top_item_ids = set(all_items_df.iloc[top_indices.cpu().numpy()][iid_col].values)

            # Check for hits
            ground_truth_items = true_positives_per_user.get(uid, set())
            
            for k in k_list:
                hits[k] += len(top_item_ids.intersection(ground_truth_items))

    recalls = {}
    for k in sorted(hits.keys()):
        recall = hits[k] / total_positive_items if total_positive_items > 0 else 0
        recalls[k] = recall
        print(f"   >>> Recall@{k}: {recall:.4f}")

    recall_at_50 = recalls.get(50, 0)
    recall_at_100 = recalls.get(100, 0)

    return recall_at_50, recall_at_100

