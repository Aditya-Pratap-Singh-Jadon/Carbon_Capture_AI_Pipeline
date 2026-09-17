"""Deterministic seed setting across Python, NumPy, and PyTorch."""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across all numerical libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
