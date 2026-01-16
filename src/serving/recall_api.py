import os
import sys
import logging
from flask import Flask, jsonify, request

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.serving.redis_storage import RedisStorage
from src.serving.rank_service import RankService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("RECSYS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RECSYS_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("RECSYS_REDIS_DB", "1"))

API_HOST = os.getenv("RECSYS_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RECSYS_API_PORT", "8000"))

app = Flask(__name__)

# Initialize Services
redis_client = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
try:
    rank_service = RankService()
    logger.info("RankService initialized.")
except Exception as e:
    logger.error(f"Failed to initialize RankService: {e}")
    rank_service = None


@app.route("/health", methods=["GET"])
def health():
    try:
        redis_client.client.ping()
        status = {"redis": "ok"}
    except Exception:
        status = {"redis": "error"}
    
    if rank_service:
        status["rank_service"] = "ok"
    else:
        status["rank_service"] = "error"
        
    return jsonify(status), 200 if status["redis"] == "ok" else 500


@app.route("/recall/<uid>", methods=["GET"])
def get_recall(uid):
    """
    Pure Recall Interface (Candidates only, no ranking)
    """
    top_k_param = request.args.get("top_k", "50")
    try:
        top_k = int(top_k_param)
    except ValueError:
        top_k = 50
    if top_k <= 0 or top_k > 500:
        top_k = 50
        
    # Convert Raw UID to Slot ID if possible
    query_uid = uid
    if rank_service:
        # feature_map keys are like "uid=123"
        slot_id = rank_service.feature_map.get(f"uid={uid}")
        if slot_id:
            query_uid = str(slot_id)
            
    items = redis_client.get_recall_results(user_id=query_uid, prefix="recall:", top_k=top_k)
    
    # Cold start fallback
    is_cold_start = False
    if not items:
        items = redis_client.get_recall_results(user_id="global_hot", prefix="", top_k=top_k)
        is_cold_start = True
        
    return jsonify({
        "uid": uid, 
        "stage": "recall",
        "count": len(items),
        "is_cold_start": is_cold_start,
        "items": items
    })


@app.route("/recommend/<uid>", methods=["GET"])
def recommend(uid):
    """
    Full Recommendation Pipeline: Recall -> Rank
    """
    # 1. Parse Args
    top_k = int(request.args.get("top_k", "10"))
    
    # 2. Recall Phase
    # Fetch more candidates for ranking (e.g., 100)
    recall_k = 100 
    
    # Convert Raw UID to Slot ID for Redis Lookup
    query_uid = uid
    if rank_service:
        slot_id = rank_service.feature_map.get(f"uid={uid}")
        if slot_id:
            query_uid = str(slot_id)
            
    candidate_items = redis_client.get_recall_results(user_id=query_uid, prefix="recall:", top_k=recall_k)
    
    is_cold_start = False
    if not candidate_items:
        candidate_items = redis_client.get_recall_results(user_id="global_hot", prefix="", top_k=recall_k)
        is_cold_start = True
    
    if not candidate_items:
        return jsonify({"uid": uid, "items": []})

    # 3. Rank Phase
    if rank_service:
        ranked_results = rank_service.predict(uid, candidate_items, top_k=top_k)
    else:
        # Fallback if ranker is down: just return recall items
        ranked_results = [{"id": iid, "score": 0.0} for iid in candidate_items[:top_k]]
        
    return jsonify({
        "uid": uid,
        "stage": "rank",
        "is_cold_start": is_cold_start,
        "items": ranked_results
    })


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)