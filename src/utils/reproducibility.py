import random
import numpy as np
import torch

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility across all relevant libraries.
    """
    if seed is not None:
        print(f"--- Setting random seed to {seed} for reproducibility ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # If using CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # for multi-GPU.
            # These settings can slow down training, but are necessary for full reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
