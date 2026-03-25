
import os
import sys
import mlflow
import pickle

# Ensure the project root is in the Python path
# This is often handled by the entrypoint script, but is good for robustness
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config, get_project_root
from src.training.trainer import train_model
from src.models.recall.dssm import DSSM
# Note: DeepFM is part of the historical code, kept for reference
# from src.models.ranking.deepfm import DeepFM 

def run_train_pipeline(config_path: str, run_name: str):
    """
    Orchestrates the model training pipeline based on the historical, stable version.
    """
    config = load_config(config_path)

    # --- MLflow Setup ---
    # MLFLOW_TRACKING_URI is set by docker-compose, so set_tracking_uri is not needed.
    mlflow.set_experiment(config['experiment_name'])

    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting run {run.info.run_id} ('{run_name}') for experiment {config['experiment_name']}")
        mlflow.log_param("config_path", config_path)
        mlflow.log_params(config)

        # --- Load Data and Metadata (Historical Method) ---
        print("\n=== Loading Data and Metadata ===")
        # The main config contains the *path* to the data config
        data_config_path = os.path.join(get_project_root(), config['data_config'])
        data_config = load_config(data_config_path)
        mlflow.log_params(data_config) # Log data params as well

        processed_dir = os.path.join(get_project_root(), data_config['processed_data_dir'])
        meta_file_path = os.path.join(processed_dir, data_config['meta_file'])
        
        print(f"Loading metadata from: {meta_file_path}")
        with open(meta_file_path, 'rb') as f:
            meta_data = pickle.load(f)

        # --- Model Initialization (Matching Historical trainer.py) ---
        print(f"\n=== Initializing Model: {config['model_name']} ===")
        
        if config['model_name'] == 'dssm':
            # This initialization matches the historical version where the model takes the full feature map
            model = DSSM(
                total_vocab_size=meta_data['feature_dims'],
                embedding_dim=config['embedding_dim'],
                hidden_dims=config['hidden_dims'],
                user_feature_count=len(meta_data['user_feature_cols']),
                item_feature_count=len(meta_data['item_feature_cols'])
                )
        # elif config['model_name'] == 'deepfm':
        #     # Placeholder for DeepFM if it needs to be restored
        #     raise NotImplementedError("DeepFM restoration not yet implemented.")
        else:
            raise ValueError(f"Unknown model name: {config['model_name']}")

        # --- Train Model ---
        # The train_model function from the user-restored trainer.py handles the full loop
        best_metric, model_path = train_model(model, config, data_config, meta_data)

        # --- Log Results to MLflow ---
        print("\n=== Logging to MLflow ===")
        # The metric name (e.g., test_auc) is defined inside trainer.py logic
        mlflow.log_metric(f"best_test_metric", best_metric)
        print(f"Logged best metric: {best_metric:.4f}")

        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")
            print(f"Logged model artifact from: {model_path}")
        else:
            print(f"Model artifact not found or not saved. Skipping artifact logging.")

    print("\nTraining pipeline finished successfully.")

