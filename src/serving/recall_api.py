import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.redis_storage import RedisStorage


REDIS_HOST = os.getenv("RECSYS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RECSYS_REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("RECSYS_REDIS_DB", "1"))

app = FastAPI()

redis_client = RedisStorage(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


class RecallResponse(BaseModel):
    uid: str
    items: list[str]


@app.get("/health")
async def health():
    try:
        redis_client.client.ping()
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=500, detail="redis_unavailable")


@app.get("/recall/{uid}", response_model=RecallResponse)
async def get_recall(uid: str, top_k: int = 50):
    if top_k <= 0 or top_k > 500:
        top_k = 50
    items = redis_client.get_recall_results(user_id=uid, prefix="recall:", top_k=top_k)
    if not items:
        items = redis_client.get_recall_results(user_id="global_hot", prefix="", top_k=top_k)
    return RecallResponse(uid=uid, items=items)

