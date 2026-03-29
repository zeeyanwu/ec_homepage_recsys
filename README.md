# End-to-End E-commerce Recommendation System

This repository contains a complete, production-ready, end-to-end e-commerce recommendation system. It demonstrates a full MLOps lifecycle, from data processing and model training to containerized deployment, model versioning, and performance testing.

## ✨ Features

- **Classic Recall-then-Rank Architecture**: A robust and scalable two-stage design.
- **Multiple Recall Strategies**:
  - Personalized Recall using **DSSM** (Deep Structured Semantic Models).
  - Global Hot List based on item popularity.
- **Advanced Ranking Model**: **DeepFM** for accurate Click-Through Rate (CTR) prediction.
- **Experiment Tracking**: Fully integrated with **MLflow** for logging models, parameters, and metrics.
- **Containerized Services**: All backend services (API, Redis, MLflow) are managed via **Docker Compose** for one-command startup and shutdown.
- **Lightweight Model Registry**: A GitOps-style model versioning system using a simple JSON file (`model_versions.json`) for promoting models to production.
- **Live Performance Monitoring**: The recommendation API logs inference latency, batch size, and unique request IDs for real-time observability.
- **Integrated Performance Testing**: Includes a dedicated script for load and stress testing the live API endpoints.

## 🏗️ Architecture Overview

The system follows a standard offline/online architecture, ensuring efficiency and scalability.

![Project Workflow](https://storage.googleapis.com/agent-ux-share/project_workflow_diagram_cn.png)

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **ML/DL**: PyTorch, Scikit-learn, Pandas
- **MLOps**: MLflow, Docker, Docker Compose
- **Database/Cache**: Redis
- **Serving**: Uvicorn

## 🚀 Getting Started

Follow these steps to set up and run the entire recommendation system on your local machine.

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zeeyanwu/ec_homepage_recsys.git
    cd ec_homepage_recsys
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch Core Infrastructure**
    This command starts the MLflow and Redis services in the background.
    ```bash
    docker-compose up -d mlflow redis
    ```

4.  **Execute the Full Workflow**
    Follow the steps in the **Workflow** section below to process data, train models, and deploy the recommendation service.

## ⚙️ Workflow: Step-by-Step Guide

This project is divided into 5 core stages. Execute them in order.

### Stage 1: Data Processing & Feature Engineering

This stage prepares the raw data into datasets suitable for model training.

- **Command**:
  ```bash
  docker-compose run --rm app python scripts/run_data_pipeline.py
  ```
- **Input**: Raw data files from `data/raw_data/`.
- **Output**:
  - Processed datasets (`train.csv`, `test.csv`, etc.) in `data/processed/`.
  - A feature mapping dictionary (`feature_map.json`).

### Stage 2: Model Training

This stage trains the recall and ranking models and logs them to MLflow.

- **Command**:
  ```bash
  # Example for training the DeepFM model
  docker-compose run --rm app python scripts/train.py --config config/deepfm.yaml
  ```
- **Action**: After a model trains successfully, copy the `Run ID` printed in the terminal.
- **Manual Step**: Paste the `Run ID` into the `production` field for the corresponding model in the `model_versions.json` file. This "promotes" the model.

### Stage 3: Offline Data Generation

This stage pre-computes recall and hot-list results and stores them in Redis for fast online lookups.

1.  **Generate Personalized Recall Lists**:
    ```bash
    # Run for each trained recall model
    docker-compose run --rm app python scripts/run_offline_recall.py --model-name dssm_inbatch
    ```
2.  **Generate Global Hot List**:
    ```bash
    docker-compose run --rm app python scripts/run_export_hot_list_to_redis.py
    ```

### Stage 4: Service Deployment & Online Recommendation

This stage launches the main `app` service, which exposes the recommendation API.

- **Command**:
  ```bash
  docker-compose up -d --force-recreate app
  ```
- **Verification**:
  - The API is now live. You can view the frontend demo at `http://localhost:8000`.
  - Check service logs for performance metrics: `docker-compose logs -f app`.

### Stage 5: Performance Evaluation

This stage runs a load test from a client's perspective to measure the API's performance.

- **Command**:
  ```bash
  # Example: Test with 5000 users at a concurrency of 50
  docker-compose run --rm app python scripts/run_stress_test.py --num-users 5000 --concurrency 50
  ```
- **Output**: A summary report detailing throughput (req/s), latency, and success rate will be printed to the console.

## 📦 Model Management

Model versions are managed via the `model_versions.json` file. This file acts as a single source of truth for which model `Run ID` is used in `production` or `staging`. To deploy a new model, simply update the `Run ID` in this file and commit the change. The services will automatically load the new model upon restart.

## <caption> Service Management

- **Start all services**:
  ```bash
  docker-compose up -d --force-recreate
  ```
- **Stop all services**:
  ```bash
  docker-compose down
  ```

## 📜 License

This project is licensed under the MIT License.
