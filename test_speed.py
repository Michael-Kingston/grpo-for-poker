import time
import torch
from pokerbot.poker_env import PokerState
from pokerbot.models import DynamicPokerLSTM
from train import collect_group_trajectories, OpponentPool

def test_speed():
    policy = DynamicPokerLSTM().to("cuda")
    env = PokerState()
    opp_pool = OpponentPool()
    
    # benchmark normal mode with ghosts (the bottleneck)
    opp_pool.mode = "normal"
    opp_pool.historical.append(DynamicPokerLSTM().to("cuda"))
    
    print("Collecting 10 groups (Normal Mode with Ghosts)...")
    start = time.time()
    for _ in range(10):
        collect_group_trajectories(policy, env, opp_pool)
    end = time.time()
    
    print(f"Time for 10 groups: {end - start:.2f}s")
    print(f"Avg time per group: {(end - start)/10:.4f}s")

if __name__ == "__main__":
    test_speed()
