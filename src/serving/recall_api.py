import os
import sys
import logging
from flask import Flask, jsonify, request, send_from_directory

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.serving.redis_storage import RedisStorage
from src.serving.rank_service import RankService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("RECSYS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RECSYS_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("RECSYS_REDIS_DB", "5"))

API_HOST = os.getenv("RECSYS_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RECSYS_API_PORT", "8000"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Static frontend
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(PROJECT_ROOT, "frontend_demo.html")

# Initialize Services
redis_client = RedisStorage(db=REDIS_DB)
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
        
    resp = jsonify(status)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp, 200 if status["redis"] == "ok" else 500


@app.route("/recommend/<uid>", methods=["GET"])
def recommend(uid):
    """
    Full Recommendation Pipeline: Multi-source Recall -> Rank
    Implements the "Unified Block, Smart Fallback, Label-assisted" architecture.
    """
    # 1. Parse Args
    top_k_display = int(request.args.get("top_k", "30"))
    if top_k_display <= 0 or top_k_display > 50:
        top_k_display = 30
    
    # Define recall sources for personalized recommendations
    personal_recall_sources = ["dssm_pointwise", "dssm_inbatch"]
    recall_k_per_source = 100 # Retrieve 100 candidates from each source

    # 2. Recall Phase - Implements Smart Fallback
    candidate_items_with_source = {} # Use dict to store {item_id: source}
    is_cold_start = False

    # -- Step 2.1: Try personalized recall first
    # Convert Raw UID to Slot ID for Redis Lookup if necessary
    query_uid = uid
    if rank_service and rank_service.feature_map:
        slot_id = rank_service.feature_map.get(f"uid={uid}")
        if slot_id:
            query_uid = str(slot_id)
            
    personal_items = redis_client.get_user_recall_results(
        user_id=query_uid,
        recall_sources=personal_recall_sources,
        top_k=recall_k_per_source
    )

    if personal_items:
        logger.info(f"Personalized recall for user {uid} found {len(personal_items)} items.")
        for item in personal_items:
            candidate_items_with_source[item] = "personal"
    else:
        # -- Step 2.2: Fallback to hot list for cold start users
        logger.info(f"Cold start for user {uid}. Falling back to global hot list.")
        is_cold_start = True
        hot_items = redis_client.get_global_hot_list(top_k=recall_k_per_source * 2) # Fetch more for fallback
        for item in hot_items:
            candidate_items_with_source[item] = "hot"

    if not candidate_items_with_source:
        logger.warning(f"No candidates found for user {uid} from any source.")
        return jsonify({"uid": uid, "items": []})

    candidate_items = list(candidate_items_with_source.keys())
    logger.info(f"Total unique candidates for ranking: {len(candidate_items)}")

    # 3. Rank Phase
    if rank_service:
        # Pass source information to ranker if it can use it as a feature
        ranked_results = rank_service.predict(uid, candidate_items, top_k=top_k_display)
    else:
        # Fallback ranking if RankService is unavailable
        logger.warning("RankService not available. Returning unranked candidates.")
        ranked_results = [{"id": iid, "score": 0.0} for iid in candidate_items[:top_k_display]]

    # 4. Add Labels
    # The `is_hot` label can be derived from the recall source
    for item in ranked_results:
        item_id_str = str(item["id"])
        item["is_hot"] = candidate_items_with_source.get(item_id_str) == "hot"

    resp = jsonify({
        "uid": uid,
        "stage": "rank",
        "is_cold_start": is_cold_start,
        "items": ranked_results
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
