import os
from flask import Flask, jsonify, request

from src.serving.redis_storage import RedisStorage
# from root import get_root_dir
# os.chdir(get_root_dir())

REDIS_HOST = os.getenv("RECSYS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RECSYS_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("RECSYS_REDIS_DB", "1"))

API_HOST = os.getenv("RECSYS_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RECSYS_API_PORT", "8000"))

app = Flask(__name__)

redis_client = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


@app.route("/health", methods=["GET"])
def health():
    try:
        redis_client.client.ping()
        return jsonify(status="ok")
    except Exception:
        return jsonify(error="redis_unavailable"), 500


@app.route("/recall/<uid>", methods=["GET"])
def get_recall(uid):
    top_k_param = request.args.get("top_k", "50")
    try:
        top_k = int(top_k_param)
    except ValueError:
        top_k = 50
    if top_k <= 0 or top_k > 500:
        top_k = 50
    items = redis_client.get_recall_results(user_id=uid, prefix="recall:", top_k=top_k)
    if not items:
        items = redis_client.get_recall_results(user_id="global_hot", prefix="", top_k=top_k)
    return jsonify(uid=uid, items=items)


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)

    # http://localhost:8000/recall/7597230350533193880?top_k=10