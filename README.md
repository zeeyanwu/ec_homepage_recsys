# Zero-query Homepage Recommendation

## 项目简介
本项目构建了一个无搜索意图的首页推荐系统，旨在提升点击率 (CTR) 和转化率 (CVR)。

### 核心模块
1.  **ETL / Data Pipeline**: 处理 shop.dat, item_feature, user_feature，生成训练数据和全局特征映射 (Slot Indexing)。
2.  **Recall Layer**: 双塔模型 (DSSM) + 热门召回 (Hot Item Recall)。
3.  **Ranking Layer**: DeepFM 多任务排序模型 (CTR/CVR)，引入 Global Score (热门度) 作为 Dense Feature。
4.  **Serving**: Redis 存储离线计算的 Recall 结果；DeepFM 用于精排。

## 环境依赖
*   Python 3.8+
*   PyTorch >= 2.0.0
*   Pandas, NumPy
*   Redis >= 4.5.0
*   Scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
请确保以下原始数据文件位于 `data/raw/` 目录：
*   `shop.dat`: 行为日志 (ts | uid | iid | label)
*   `item_feature.dat`: 商品特征 (iid | tag1 | tag2 | tag3)
*   `user_feature.dat`: 用户特征 (uid | tag1 | tag2)

## 运行步骤 (Manual Execution)

### 1. ETL 数据预处理
执行预处理脚本，生成训练集、测试集、Meta Data 和 Global Scores。
```bash
python src/etl/preprocessor.py
```
*   输出目录: `data/processed/`
*   关键产出: `train.csv`, `test.csv`, `meta_data.pkl`, `item_global_score.csv`

### 2. 训练 Recall 模型 (DSSM)
训练双塔召回模型，并评估 Recall@K 指标。
```bash
python src/training/train_recall_dssm.py
```
*   模型保存: `src/models/saved/dssm_inbatch.pth`

### 3. 导出 Recall 结果到 Redis
将计算好的 User Recall 结果 (Top-K Items) 写入 Redis。
(请确保本地 Redis 服务已启动: `redis-server`)
```bash
python src/serving/export_to_redis.py --host localhost --port 6379
```
*   Redis Key: `dssm:recall:{user_slot_id}`

### 4. 训练 Ranking 模型 (DeepFM)
训练 DeepFM 排序模型，并评估 AUC 指标。
```bash
python src/training/train_rank_deepfm.py
```
*   模型保存: `src/models/saved/deepfm.pth`

### 5. (Optional) 验证 Recall 服务接口
测试 Redis 读取接口。
```bash
python src/serving/recall_service.py
```

## 架构说明
*   **Feature Engineering**: 使用 Slot-based 全局特征编码。
*   **Recall**: 
    *   Offline Batch Prediction: `export_to_redis.py` 计算 User Embedding 与 All Item Embeddings 的相似度，存入 Redis。
    *   Online Serving: 直接读取 Redis 中的 Item List。
*   **Ranking**:
    *   DeepFM 接收 Recall 的候选集。
    *   特征: User/Item ID (Sparse/Shared Embedding) + Global Score (Dense)。
