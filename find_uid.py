import pickle
import os
import sys

# Add project root
sys.path.append(os.getcwd())

def find_raw_uid(target_slot_id):
    map_path = 'data/processed/feature_map.pkl'
    if not os.path.exists(map_path):
        print("Feature map not found.")
        return
        
    print(f"Loading feature map from {map_path}...")
    with open(map_path, 'rb') as f:
        feature_map = pickle.load(f)
        
    # feature_map structure: {'uid=123': 1, 'iid=abc': 2, ...}
    # We need to reverse lookup: value -> key
    
    target_key = None
    for key, slot_id in feature_map.items():
        if slot_id == target_slot_id:
            # key format is usually "uid=xxxx"
            if key.startswith("uid="):
                target_key = key
                break
    
    if target_key:
        raw_uid = target_key.split("=", 1)[1]
        print(f"\nFOUND! Slot ID {target_slot_id} corresponds to Raw User ID: {raw_uid}")
        print(f"Test URL: http://localhost:8000/recommend/{raw_uid}?top_k=10")
    else:
        print(f"\nSlot ID {target_slot_id} not found in feature map or is not a user ID.")

if __name__ == "__main__":
    # The slot id user found in Redis
    find_raw_uid(618)
