from pokerbot.poker_env import PokerState
from pokerbot.models import DynamicPokerLSTM
from pokerbot.config import DEVICE

def test_snapshot():
    print("Testing Snapshotting...")
    env = PokerState()
    env.reset()
    
    # steps
    env.step(0, 1, 0.0) 
    
    # save
    snapshot = env.save_state()
    
    # more steps
    env.step(1, 2, 0.5) 
    stack_before = env.stacks[0]
    
    # load
    env.load_state(snapshot)
    
    # verify
    assert env.round == snapshot.round
    assert env.stacks[0] == snapshot.stacks[0]
    assert env.chips_in_front[0] == snapshot.chips_in_front[0]
    print("Snapshot restoration SUCCESS")

def test_features():
    print("Testing Features...")
    env = PokerState()
    env.reset()
    obs = env.get_obs(0)
    
    # shape check
    print(f"Obs shape: {obs.shape}")
    assert obs.shape[0] == 135 + 20 
    
    # spr
    spr = obs[131]
    print(f"Initial SPR: {spr.item()}")
    assert spr > 0
    
    # mask
    mask = obs[132:135]
    print(f"Initial Mask: {mask}")
    assert mask.sum() > 0
    print("Feature Engineering SUCCESS")

def test_temperature():
    print("Testing Temperature...")
    policy = DynamicPokerLSTM().to(DEVICE)
    env = PokerState()
    env.reset()
    obs = env.get_obs(0)
    mask = env.get_mask(0)
    
    # action runs
    c, a, lp = policy.get_action(obs, mask, temperature=2.0)
    print(f"Action with Temp 2.0: {c}, {a}")
    
    c, a, lp = policy.get_action(obs, mask, temperature=0.1)
    print(f"Action with Temp 0.1: {c}, {a}")
    print("Temperature implementation SUCCESS")

if __name__ == "__main__":
    test_snapshot()
    test_features()
    test_temperature()
    print("\nALL TESTS PASSED")
