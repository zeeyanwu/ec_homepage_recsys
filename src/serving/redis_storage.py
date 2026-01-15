import redis
import json
import numpy as np

class RedisStorage:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        
    def save_recall_results(self, user_id, item_ids, scores=None, prefix='recall:'):
        """
        保存召回结果到 Redis
        Key: recall:{user_id}
        Value: List of item_ids (or json string)
        """
        key = f"{prefix}{user_id}"
        # 简单存储为 JSON 字符串，或者使用 Redis List / Sorted Set
        # 这里使用 List
        self.client.delete(key)
        if scores:
            # 使用 Sorted Set (ZSET) 存储 (item_id, score)
            data = {item: float(score) for item, score in zip(item_ids, scores)}
            self.client.zadd(key, data)
        else:
            self.client.rpush(key, *item_ids)
            
    def get_recall_results(self, user_id, prefix='recall:', top_k=50):
        key = f"{prefix}{user_id}"
        type_ = self.client.type(key)
        
        if type_ == 'zset':
            items = self.client.zrevrange(key, 0, top_k - 1)
        elif type_ == 'list':
            items = self.client.lrange(key, 0, top_k - 1)
        else:
            items = []
        return items
        
    def save_item_embedding(self, item_id, vector, prefix='item_emb:'):
        """
        保存 Item Embedding 用于向量检索 (FAISS or RedisSearch)
        """
        key = f"{prefix}{item_id}"
        # Store as bytes or json list
        self.client.set(key, json.dumps(vector.tolist()))

    def get_item_embedding(self, item_id, prefix='item_emb:'):
        key = f"{prefix}{item_id}"
        data = self.client.get(key)
        if data:
            return np.array(json.loads(data))
        return None
