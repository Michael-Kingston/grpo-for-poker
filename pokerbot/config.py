import torch

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grpo config 
GROUP_SIZE = 64
BATCH_SIZE = 2048
PPO_EPOCHS = 10
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.05
LR = 3e-4
HIDDEN_DIM = 256

# obs config
MAX_HISTORY = 20
STATIC_DIM = 135
TOTAL_OBS_DIM = STATIC_DIM + MAX_HISTORY

# game config
START_STACK = 100.0
MIN_RAISE = 2.0
