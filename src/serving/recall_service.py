import redis
import json
import os

class RecallService:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        
    def get_recall_items(self, user_id, top_k=50):
        """
        Get recall items for a user.
        Returns personalized DSSM recall results only.
        
        Args:
            user_id (str): The User ID (Slot ID as used in training)
            top_k (int): Number of items to return
            
        Returns:
            list: List of Item IDs (Slot IDs)
        """
        # 1. Get Personalized Recall (DSSM)
        # Key format matches export_to_redis.py: "dssm:recall:{uid}"
        dssm_key = f"dssm:recall:{user_id}"
        dssm_items = self.client.zrevrange(dssm_key, 0, top_k - 1)
        
        return dssm_items

if __name__ == "__main__":
    # Test
    try:
        service = RecallService()
        # Test with a dummy user id '0' (assuming slot 0 exists)
        items = service.get_recall_items(user_id='0', top_k=10)
        print(f"Recall results for User 0: {items}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Redis is running and data is exported.")
