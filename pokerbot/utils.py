import random
import numpy as np
import torch
from collections import namedtuple

# data structure for trajectory steps
StepData = namedtuple('StepData', ['obs', 'mask', 'cat_action', 'amt_action', 'log_prob', 'reward'])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
