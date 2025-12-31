import torch

# device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grpo config - reduced for speed on weak hardware
GROUP_SIZE = 32   # was 128
BATCH_SIZE = 64   # was 256
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.02
LR = 3e-4

# observation config
MAX_HISTORY = 10
STATIC_DIM = 9
TOTAL_OBS_DIM = STATIC_DIM + MAX_HISTORY

# game config
START_STACK = 20.0
MIN_RAISE = 2.0
