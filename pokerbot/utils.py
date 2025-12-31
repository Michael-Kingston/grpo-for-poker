import random
import numpy as np
import torch
from collections import namedtuple

# data structure for trajectory steps
StepData = namedtuple('StepData', ['obs', 'mask', 'cat_action', 'amt_action', 'log_prob', 'reward'])

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def card_to_str(card):
    if card is None: return "None"
    ranks = "23456789TJQKA"
    suits = "shdc"
    r, s = card // 4, card % 4
    return f"{ranks[r]}{suits[s]}"
