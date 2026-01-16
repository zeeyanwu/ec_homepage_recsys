# Zero-query Homepage Recommendation

End-to-end zero-query homepage recommendation system for e-commerce, targeting CTR and CVR uplift on the home feed when users arrive **without an explicit search query**.

The system implements a full offline–online pipeline:

- ETL & feature engineering
- Recall (DSSM) + hot item recall
- Ranking (DeepFM with global hotness as dense feature)
- Online serving via Redis + Flask API

---

## 1. Project Overview

### Core Modules

1. **ETL / Data Pipeline**
   - Reads raw logs and features (`shop.dat`, `user_feature.dat`, `item_feature.dat`) from `data/raw_data/`
   - Builds a **global slot-based feature map** (string key `col=value` → integer ID)
   - Generates train/test CSVs for both recall and ranking
   - Computes **global item hotness score** with time decay + Bayesian-smoothed CTR

2. **Recall Layer (DSSM)**
   - Dual-tower semantic model (user tower + item tower)
   - Supports **pointwise** training with negative sampling and **in-batch** negative training
   - Offline batch scoring to generate Top-K candidates per user

3. **Ranking Layer (DeepFM)**
   - DeepFM CTR model for fine-grained ranking of recall candidates
   - Sparse features: user & item IDs and categorical tags
   - Dense feature: global item hotness score from ETL

4. **Serving Layer**
   - Redis stores merged recall lists and global hot items
   - Flask app exposes HTTP APIs for:
     - Pure recall candidates
     - Full recommendation pipeline: Recall → Rank

---

## 2. Environment & Dependencies

### Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Pandas, NumPy
- Scikit-learn
- Redis >= 4.5.0 (running locally or remotely)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Data Preparation

Place the following raw data files under `data/raw_data/`:

- `shop.dat`  
  Interaction log:  
  `ts | uid | iid | label`
- `user_feature.dat`  
  User features:  
  `uid | utag1 | utag2`
- `item_feature.dat`  
  Item features:  
  `iid | itag1 | itag2 | itag3`

> `uid` / `iid` here are **raw IDs** from the data provider. They will be mapped into global slot IDs during ETL.

---

## 4. Offline Pipeline: Step-by-step

All commands assume you are in the project root:

```bash
cd /path/to/ec_homepage_recsys
```

### 4.1 ETL & Feature Engineering

Run the ETL preprocessor to:

- Merge logs and features
- Build global feature map (slot indexing)
- Generate `train.csv`, `test.csv`
- Compute global item hotness scores

```bash
python src/etl/preprocessor.py
```

Outputs (under `data/processed/`):

- `train.csv`
- `test.csv`
- `meta_data.pkl` (feature dimensions and column config)
- `feature_map.pkl` (global slot mapping: `"col=value" → int`)
- `item_global_score.csv` (includes `iid`, `slot_id`, `global_score`, `impression_count`, `ctr_mean`)

### 4.2 Train Recall Model (DSSM)

The project provides two training modes for DSSM:

1. **Pointwise training with negative sampling**  
   Uses BCE loss with explicit positive/negative samples.

   ```bash
   python src/training/train_recall_dssm_pointwise.py
   ```

   Model is saved as:

   - `src/models/saved/dssm_pointwise.pth`

2. **In-batch negative training** (optional / experimental)

   ```bash
   python src/training/train_recall_dssm_inbatch.py
   ```

   Model is saved as:

   - `src/models/saved/dssm_inbatch.pth`

The default downstream export script uses the **pointwise** model (`dssm_pointwise.pth`).

### 4.3 Train Ranking Model (DeepFM)

Train the DeepFM CTR ranking model using the processed data and global hotness score:

```bash
python src/training/train_rank_deepfm.py
```

This script:

- Loads `train.csv` / `test.csv` and `meta_data.pkl`
- Builds sparse inputs from user/item slot IDs
- Builds a dense feature from `item_global_score.csv` (per-item global_score)
- Trains DeepFM and reports AUC

Model is saved as:

- `src/models/saved/deepfm.pth`

### 4.4 Export Recall Results & Global Hot to Redis

Before running export, make sure a Redis instance is available, for example:

```bash
redis-server
```

Then run:

```bash
python src/serving/export_to_redis.py --host localhost --port 6379
```

This script will:

1. Load:
   - `meta_data.pkl`
   - `train.csv`, `test.csv`
   - `dssm_pointwise.pth`
   - `item_global_score.csv`
2. Compute user–item similarity scores using the DSSM model
3. Merge DSSM Top-K candidates per user with global hot items using an interleaving strategy
4. Write the following keys to Redis (DB = 1 by default):

- Per-user merged recall list:
  - Key pattern: `recall:{user_slot_id}`
  - Value: ZSET of **raw item IDs** with descending scores encoding order
- Global hot items:
  - Key: `global_hot`
  - Value: ZSET of **raw item IDs** with their `global_score`

These keys will be consumed by the online serving layer.

---

## 5. Online Serving

### 5.1 Components

- **Redis**: stores merged recall lists and global hot items (as above)
- **RankService**: wraps DeepFM model and feature lookup for inference  
  Location: `src/serving/rank_service.py`
- **Flask API**: main HTTP entrypoint  
  Location: `src/serving/recall_api.py`

### 5.2 Environment Variables

The serving app reads the following environment variables (with defaults):

- `RECSYS_REDIS_HOST` (default: `localhost`)
- `RECSYS_REDIS_PORT` (default: `6379`)
- `RECSYS_REDIS_DB` (default: `1`)
- `RECSYS_API_HOST` (default: `0.0.0.0`)
- `RECSYS_API_PORT` (default: `8000`)

### 5.3 Start the API Server

From the project root:

```bash
python src/serving/recall_api.py
```

You should see logs similar to:

- DeepFM model loaded
- RankService initialized
- Flask app listening on `http://0.0.0.0:8000`

### 5.4 API Endpoints

#### 5.4.1 Health Check

```bash
GET /health
```

Response example:

```json
{
  "redis": "ok",
  "rank_service": "ok"
}
```

#### 5.4.2 Pure Recall (Candidates Only)

```bash
GET /recall/<uid>?top_k=50
```

- `uid`: **raw user ID** from the original data
- Internally:
  - The service converts raw `uid` → slot ID using the feature map loaded in `RankService`
  - Uses the slot ID to look up `recall:{user_slot_id}` in Redis
  - If no record exists, falls back to `global_hot`

Response example:

```json
{
  "uid": "5905843863414246198",
  "stage": "recall",
  "count": 50,
  "is_cold_start": false,
  "items": [
    "2913544956788485329",
    "18292817638588633000",
    "... more item ids ..."
  ]
}
```

#### 5.4.3 Full Recommendation: Recall → Rank

```bash
GET /recommend/<uid>?top_k=10
```

- `uid`: raw user ID
- `top_k`: number of ranked items to return (default 10)

Pipeline inside the endpoint:

1. Convert raw `uid` → slot ID
2. Fetch up to 100 candidates from `recall:{user_slot_id}` (or `global_hot` for cold-start)
3. Use `RankService` (DeepFM) to compute CTR scores for each candidate
4. Sort by score and return the top `top_k`

Response example:

```json
{
  "uid": "5905843863414246198",
  "stage": "rank",
  "is_cold_start": false,
  "items": [
    { "id": "2913544956788485329", "score": 0.99 },
    { "id": "18292817638588633000", "score": 0.99 },
    { "id": "16199685692793916813", "score": 0.98 }
  ]
}
```

---

## 6. High-level Architecture Summary

- **Feature Engineering**
  - Global slot-based encoding ensures consistent IDs across recall and ranking
  - Global hotness score combines **popularity (impressions)** and **CTR** with time decay and Bayesian smoothing

- **Recall**
  - DSSM computes dense embeddings for users and items
  - Offline batch scoring writes Top-K raw item IDs per user into Redis (`recall:{uid_slot}`)
  - Global hot items are exported separately for cold-start users (`global_hot`)

- **Ranking**
  - DeepFM takes:
    - Sparse features: slot IDs for user and item
    - Dense feature: `global_score` from ETL
  - Outputs CTR probability used for sorting

- **Serving**
  - `/recall/<uid>`: view recall results only (debugging / analysis)
  - `/recommend/<uid>`: full pipeline (Recall → Rank) used by the homepage feed

This README reflects the current end-to-end implementation of the **zero-query homepage recommendation** system, including ETL, recall (DSSM), ranking (DeepFM), Redis export, and online serving via Flask.
