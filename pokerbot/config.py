import torch

# device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grpo config
GROUP_SIZE = 32
BATCH_SIZE = 64
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.02
LR = 3e-4

# obs config
MAX_HISTORY = 20 # round history
STATIC_DIM = 131 # hole + board + numeric
TOTAL_OBS_DIM = STATIC_DIM + MAX_HISTORY

# game config
START_STACK = 100.0
MIN_RAISE = 2.0
