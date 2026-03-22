
import os
import sys
import mlflow
import pickle
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, get_project_root
from src.training.trainer import train_model
from src.models.ranking.deepfm import DeepFM
from src.models.recall.dssm import DSSM


def run_train_pipeline(config):
    """
    Orchestrates the model training pipeline.

    1. Sets up MLflow tracking.
    2. Loads data configuration and metadata.
    3. Initializes the appropriate model based on the config.
    4. Starts the training process.
    5. Logs parameters, metrics, and the model artifact to MLflow.
    """
    # --- MLflow Setup ---
    mlflow.set_tracking_uri(f"file://{os.path.join(get_project_root(), 'mlruns')}")
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run() as run:
        print(f"Starting run {run.info.run_id} for experiment {config['experiment_name']}")
        # Log all config parameters
        mlflow.log_params(config)

        # --- Load Data and Metadata ---
        print("\n=== Loading Data and Metadata ===")
        data_config = load_config(config['data_config'])
        mlflow.log_params(data_config) # Log data params as well

        root_dir = get_project_root()
        processed_dir = os.path.join(root_dir, data_config['processed_data_dir'])

        with open(os.path.join(processed_dir, data_config['meta_file']), 'rb') as f:
            meta_data = pickle.load(f)

        # --- Model Initialization ---
        print(f"\n=== Initializing Model: {config['model_name']} ===")
        model = None
        if config['model_name'] == 'deepfm':
            model = DeepFM(
                total_vocab_size=meta_data['feature_dims'],
                num_sparse_features=len(meta_data['user_feature_cols']) + len(meta_data['item_feature_cols']),
                num_dense_features=1,  # global_score
                embedding_dim=config['embedding_dim'],
                hidden_dims=[64, 32], # TODO: Add to config if needed
                dropout=0.5 # TODO: Add to config if needed
            )
        elif config['model_name'] == 'dssm':
            model = DSSM(
                total_vocab_size=meta_data['feature_dims'],
                embedding_dim=config['embedding_dim'],
                hidden_dims=config['hidden_dims'],
                user_feature_count=len(meta_data['user_feature_cols']),
                item_feature_count=len(meta_data['item_feature_cols'])
            )
        else:
            raise ValueError(f"Unknown model name: {config['model_name']}")

        if model:
            # --- Train Model ---
            # The train_model function will be responsible for the actual training loop and evaluation
            best_metric, model_path = train_model(model, config, data_config, meta_data)

            # --- Log Results to MLflow ---
            print("\n=== Logging to MLflow ===")
            mlflow.log_metric(f"best_test_metric", best_metric)
            print(f"Logged best metric: {best_metric:.4f}")

            # Log the model as an artifact, if it exists
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="model")
                print(f"Logged model artifact from: {model_path}")
            else:
                print(f"No model was saved, as performance did not improve. Skipping artifact logging.")

        print("\nTraining pipeline finished successfully.")

