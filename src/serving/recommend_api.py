import os
import sys
import logging
import time
import uuid
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
    request_id = str(uuid.uuid4())
    # 1. Parse Args
    top_k_display = int(request.args.get("top_k", "30"))
    if top_k_display <= 0 or top_k_display > 50:
        top_k_display = 30
    
    # 2. Recall Phase - Multi-source Fusion with Cold Start Logic
    candidate_items_with_source = {}  # Use dict to store {item_id: source}

    # -- Define recall sources and tunable sizes
    personal_recall_sources = ["dssm_pointwise", "dssm_inbatch"]
    PERSONAL_K = 100  # K for each personal source (pointwise, in-batch)
    HOT_K_ACTIVE_USER = 50  # K for hot list for an active user
    HOT_K_COLD_START = 200  # K for hot list for a cold-start user

    # -- Step 2.1: Fetch personalized recall using raw UID
    personal_items = redis_client.get_user_recall_results(
        user_id=uid, # Use raw UID directly
        recall_sources=personal_recall_sources,
        top_k=PERSONAL_K
    )
    for item in personal_items:
        candidate_items_with_source[item] = "personal"

    # -- Step 2.2: Fetch global hot list, adjusting K based on user type (cold/active)
    is_cold_start = not bool(personal_items)
    if is_cold_start:
        logger.info(f"Cold start for user {uid}. Fetching {HOT_K_COLD_START} items from global hot list.")
        hot_items = redis_client.get_global_hot_list(top_k=HOT_K_COLD_START)
    else:
        logger.info(f"Active user {uid}. Found {len(personal_items)} personal items. Fusing with {HOT_K_ACTIVE_USER} hot items.")
        hot_items = redis_client.get_global_hot_list(top_k=HOT_K_ACTIVE_USER)
    
    for item in hot_items:
        candidate_items_with_source[item] = "hot"

    if not candidate_items_with_source:
        logger.warning(f"No candidates found for user {uid} from any source.")
        return jsonify({"uid": uid, "items": []})

    candidate_items = list(candidate_items_with_source.keys())
    batch_size = len(candidate_items)
    logger.info(f"Total unique candidates for ranking for request_id={request_id}: {batch_size}")

    # 3. Rank Phase
    if rank_service:
        # Pass source information to ranker if it can use it as a feature
        start_time = time.time()
        ranked_results = rank_service.predict(uid, candidate_items, top_k=top_k_display)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        logger.info(
            f"[PERF] request_id={request_id} batch_size={batch_size} inference_time={inference_time_ms:.2f}ms"
        )
    else:
        # Fallback ranking if RankService is unavailable
        logger.warning(f"RankService not available for request_id={request_id}. Returning unranked candidates.")
        ranked_results = [{"id": iid, "score": 0.0} for iid in candidate_items[:top_k_display]]

    # 4. Add Labels
    # The `is_hot` label can be derived from the recall source
    for item in ranked_results:
        item_id_str = str(item["id"])
        item["is_hot"] = candidate_items_with_source.get(item_id_str) == "hot"

    resp = jsonify({
        "uid": uid,
        "request_id": request_id,
        "stage": "rank",
        "is_cold_start": is_cold_start,
        "items": ranked_results
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
