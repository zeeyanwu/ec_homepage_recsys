import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.ranking.deepfm import DeepFM

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RankService:
    def __init__(self, model_dir='src/models/saved', data_dir='data/processed', raw_data_dir='data/raw_data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        
        self.device = torch.device('cpu') # Inference on CPU usually fine for this scale
        
        # 1. Load Metadata & Feature Map
        self._load_metadata()
        
        # 2. Load Feature Store (User/Item Features)
        self._load_feature_store()
        
        # 3. Load Model
        self._load_model()
        
        logger.info("RankService initialized successfully.")

    def _load_metadata(self):
        # Load Meta Data
        meta_path = os.path.join(self.data_dir, 'meta_data.pkl')
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            
        self.feature_dims = self.meta['feature_dims']
        self.user_cols = self.meta['user_feature_cols']
        self.item_cols = self.meta['item_feature_cols']
        
        # Load Feature Map (Key -> Slot ID)
        map_path = os.path.join(self.data_dir, 'feature_map.pkl')
        with open(map_path, 'rb') as f:
            self.feature_map = pickle.load(f)
            
        # Load Global Scores
        score_path = os.path.join(self.data_dir, 'item_global_score.csv')
        self.global_score_map = {}
        if os.path.exists(score_path):
            df_score = pd.read_csv(score_path)
            # Map raw iid to global_score (Note: CSV has 'iid', 'slot_id', 'global_score')
            # We use raw iid for lookup during inference
            self.global_score_map = dict(zip(df_score['iid'].astype(str), df_score['global_score']))

    def _load_feature_store(self):
        # Simple Memory Feature Store
        # Load User Features
        user_path = os.path.join(self.raw_data_dir, 'user_feature.dat')
        # names=['uid', 'utag1', 'utag2']
        df_user = pd.read_csv(user_path, header=None, names=['uid', 'utag1', 'utag2'])
        df_user['uid'] = df_user['uid'].astype(str)
        # Convert to dict for O(1) lookup: uid -> {col: val}
        self.user_features = df_user.set_index('uid').to_dict('index')
        
        # Load Item Features
        item_path = os.path.join(self.raw_data_dir, 'item_feature.dat')
        # names=['iid', 'itag1', 'itag2', 'itag3']
        df_item = pd.read_csv(item_path, header=None, names=['iid', 'itag1', 'itag2', 'itag3'])
        df_item['iid'] = df_item['iid'].astype(str)
        self.item_features = df_item.set_index('iid').to_dict('index')
        
        # Default features if missing
        self.default_user_feat = {col: '0' for col in self.user_cols if col != 'uid'}
        self.default_item_feat = {col: '0' for col in self.item_cols if col != 'iid'}

    def _load_model(self):
        self.model = DeepFM(
            total_vocab_size=self.feature_dims,
            num_sparse_features=len(self.user_cols) + len(self.item_cols),
            num_dense_features=1,
            embedding_dim=16,
            hidden_dims=[64, 32],
            dropout=0.0
        )
        
        model_path = os.path.join(self.model_dir, 'deepfm.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded DeepFM model from {model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

    def get_slot_id(self, col, val):
        key = f"{col}={val}"
        return self.feature_map.get(key, 0) # 0 is usually safe or we should use a specific unknown index?
        # In preprocessor, we didn't reserve 0 for unknown explicitly, but index starts at 0.
        # If unknown, maybe mapped to something else? 
        # For now, let's assume 0 or handle unknowns better if we had an UNK token.
        # Given preprocessor logic: if key not in map, it wasn't in training.
        # Ideally we should have an UNK bucket. But here we'll just return 0 (collision risk but simple).
        
    def predict(self, user_id, item_ids, top_k=50):
        """
        Rank a list of items for a user.
        """
        if not item_ids:
            return []
            
        user_id = str(user_id)
        item_ids = [str(iid) for iid in item_ids]
        
        # 1. Prepare Features
        # User side (broadcast)
        u_feat_raw = self.user_features.get(user_id, self.default_user_feat)
        
        # 2. Build Batch Data
        sparse_indices = []
        dense_values = []
        
        valid_items = []
        
        for iid in item_ids:
            i_feat_raw = self.item_features.get(iid, self.default_item_feat)
            
            # Combine User + Item Features
            # Order must match self.user_cols + self.item_cols
            current_slots = []
            
            # User Features
            for col in self.user_cols:
                if col == 'uid':
                    val = user_id
                else:
                    val = u_feat_raw.get(col, '0')
                current_slots.append(self.get_slot_id(col, val))
                
            # Item Features
            for col in self.item_cols:
                if col == 'iid':
                    val = iid
                else:
                    val = i_feat_raw.get(col, '0')
                current_slots.append(self.get_slot_id(col, val))
                
            sparse_indices.append(current_slots)
            
            # Dense Feature (Global Score)
            g_score = self.global_score_map.get(iid, 0.0)
            dense_values.append([g_score])
            valid_items.append(iid)
            
        # 3. To Tensor
        sparse_tensor = torch.tensor(sparse_indices, dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor(dense_values, dtype=torch.float32, device=self.device)
        
        # 4. Inference
        with torch.no_grad():
            scores = self.model(sparse_tensor, dense_tensor).squeeze().cpu().numpy()
            
        # Handle single item case (scalar output)
        if scores.ndim == 0:
            scores = [float(scores)]
        else:
            scores = scores.tolist()
            
        # 5. Sort
        # item_scores = list(zip(valid_items, scores))
        # item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        # return item_scores[:top_k]
        
        # Return dict or list? Let's return list of dicts
        results = []
        for iid, score in zip(valid_items, scores):
            results.append({"id": iid, "score": float(score)})
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    # Test Run
    ranker = RankService()
    
    # Real IDs from head of data files
    test_uid = "7597230350533193880"
    test_items = [
        "1083856320208763226", 
        "17503324300351289023",
        "3576283427002404729",
        "9999999999" # Fake
    ]
    
    print(f"Ranking for User {test_uid}:")
    results = ranker.predict(test_uid, test_items)
    for res in results:
        print(res)
