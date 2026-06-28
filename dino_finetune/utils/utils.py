import torch
import numpy as np
import random

def fix_seeds(seed: int = 3407) -> None:
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)