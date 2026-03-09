import yaml
import os

def load_config(config_path):
    """
    Loads a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_project_root():
    """
    Dynamically computes the project root directory.
    Assumes this file is in src/utils/
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
