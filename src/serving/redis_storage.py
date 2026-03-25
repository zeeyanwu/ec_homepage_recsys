import redis
import json
import numpy as np
import os

class RedisStorage:
    def __init__(self, db=5, password=None):
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", 6379))
        password = os.environ.get("REDIS_PASSWORD", password)
        self.client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)

    def save_user_recall_results(self, user_id, item_ids, recall_source):
        """
        保存指定来源的【个性化】召回结果到 Redis List.
        Key: recall:{recall_source}:{user_id}
        e.g., recall:dssm_pointwise:123
        """
        key = f"recall:{recall_source}:{user_id}"
        # 使用 pipeline 提高效率
        pipe = self.client.pipeline()
        pipe.delete(key)
        if item_ids:
            pipe.rpush(key, *item_ids)
        pipe.execute()

    def get_user_recall_results(self, user_id, recall_sources, top_k):
        """
        从多个来源获取【个性化】召回结果，并在线融合.
        """
        all_items = set()
        
        # 使用 pipeline 并行获取多路召回结果
        pipe = self.client.pipeline()
        for source in recall_sources:
            key = f"recall:{source}:{user_id}"
            pipe.lrange(key, 0, top_k - 1)
        
        results = pipe.execute()
        
        for items in results:
            if items:
                all_items.update(items)
                
        return list(all_items)

    def save_global_hot_list(self, item_scores: dict, list_name='global_hot'):
        """
        保存全局热榜到 Redis Sorted Set.
        Key: recall:hot_list:{list_name}
        """
        key = f"recall:hot_list:{list_name}"
        pipe = self.client.pipeline()
        pipe.delete(key)
        if item_scores:
            pipe.zadd(key, item_scores)
        pipe.execute()

    def get_global_hot_list(self, list_name='global_hot', top_k=200):
        """
        从 Redis Sorted Set 获取全局热榜.
        """
        key = f"recall:hot_list:{list_name}"
        # ZREVRANGE to get items with highest scores
        items_with_scores = self.client.zrevrange(key, 0, top_k - 1, withscores=True)
        # 只返回 item_id
        return [item for item, score in items_with_scores]

    def save_item_embedding(self, item_id, vector, prefix='item_emb:'):
        """
        保存 Item Embedding 用于向量检索 (FAISS or RedisSearch)
        """
        key = f"{prefix}{item_id}"
        self.client.set(key, json.dumps(vector.tolist()))

    def get_item_embedding(self, item_id, prefix='item_emb:'):
        key = f"{prefix}{item_id}"
        data = self.client.get(key)
        if data:
            return np.array(json.loads(data))
        return None
