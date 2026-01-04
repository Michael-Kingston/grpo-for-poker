import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time
import torch.nn.functional as F
from pokerbot.config import DEVICE, STATIC_DIM
from pokerbot.models import DynamicPokerLSTM
from pokerbot.poker_env import PokerState
from pokerbot.utils import card_to_str

def print_game_state(env, human_idx=1):
    print("\n" + "="*40)
    # state
    board_str = " ".join([card_to_str(c) for c in env.board])
    if not board_str: board_str = "empty"
    print(f"POT: {env.pot:.2f}  |  BOARD: {board_str}")
    print("-" * 40)
    
    # stacks
    bot_idx = 1 - human_idx
    print(f"BOT Stack: {env.stacks[bot_idx]:.2f}  |  In Front: {env.chips_in_front[bot_idx]:.2f}")
    print(f"YOU Stack: {env.stacks[human_idx]:.2f}  |  In Front: {env.chips_in_front[human_idx]:.2f}")
    
    # hand
    hand_str = " ".join([card_to_str(c) for c in env.cards[human_idx]])
    print(f"YOUR HAND: [{hand_str}]")
    print("="*40)

def get_human_action(env, p_idx):
    mask = env.get_mask(p_idx).cpu().numpy()
    
    print("\nOptions:")
    options = []
    if mask[0] == 1: options.append("f (Fold)")
    if mask[1] == 1: options.append("c (Call/Check)")
    if mask[2] == 1: options.append("b (Bet/Raise)")
    
    print(" | ".join(options))
    
    while True:
        choice = input("Your Move: ").strip().lower()
        if choice == 'f' and mask[0] == 1: return 0, 0.0
        if choice == 'c' and mask[1] == 1: return 1, 0.0
        if choice == 'b' and mask[2] == 1:
            opp_idx = 1 - p_idx
            curr_opp_bet = env.chips_in_front[opp_idx]
            my_current = env.chips_in_front[p_idx]
            min_raise_const = 2.0
            
            call_cost = max(0, curr_opp_bet - my_current)
            available = max(0, env.stacks[p_idx] - call_cost)
            
            min_bet_amt = min(min_raise_const, available)
            max_bet_amt = available
            
            print(f"Bet Amount (Min {min_bet_amt:.2f} - Max {max_bet_amt:.2f}):")
            try:
                amt_input = float(input("> "))
            except ValueError: continue
            if amt_input < min_bet_amt or amt_input > max_bet_amt: continue
                
            # frac
            frac = 1.0 if max_bet_amt - min_bet_amt < 1e-9 else (amt_input - min_bet_amt) / (max_bet_amt - min_bet_amt)
            return 2, frac
        print("Invalid choice.")

def play():
    print("loading model...")
    policy = DynamicPokerLSTM().to(DEVICE)
    try:
        policy.load_state_dict(torch.load("poker_model.pt", map_location=DEVICE))
        policy.eval()
    except FileNotFoundError:
        print("model not found. train first.")
        return

    env = PokerState()
    human_idx = 1
    bot_idx = 0
    current_stacks = [100.0, 100.0]
    
    while True:
        if current_stacks[0] < 1.0 or current_stacks[1] < 1.0:
            print("\n!!! GAME OVER !!!")
            break

        env.reset(starting_stacks=current_stacks)
        print(f"\n\n>>> NEW HAND <<<")
        curr_turn = 0
        
        while not env.finished:
            is_human = (curr_turn == human_idx)
            obs = env.get_obs(curr_turn)
            mask = env.get_mask(curr_turn)
            
            if is_human:
                print_game_state(env, human_idx)
                cat, amt = get_human_action(env, human_idx)
                env.step(human_idx, cat, amt)
            else:
                time.sleep(0.5) 
                with torch.no_grad():
                    cat, amt, _ = policy.get_action(obs, mask)
                env.step(bot_idx, cat, amt)
                print(f"\n> BOT chooses: {['Fold', 'Call', 'Bet'][cat]}")
            
            curr_turn = 1 - curr_turn
            
        print("\n--- ROUND OVER ---")
        if env.winner_idx == -1: env.resolve_showdown()
             
        payoff_human = env.get_payoff(human_idx)
        current_stacks[human_idx] = env.stacks[human_idx] + (env.pot if env.winner_idx == human_idx else (env.pot/2 if env.winner_idx == -1 else 0))
        current_stacks[bot_idx] = env.stacks[bot_idx] + (env.pot if env.winner_idx == bot_idx else (env.pot/2 if env.winner_idx == -1 else 0))
        
        print(f"winner: {['BOT', 'YOU'][env.winner_idx] if env.winner_idx != -1 else 'SPLIT'}")
        print(f"bot hand: {' '.join([card_to_str(c) for c in env.cards[bot_idx]])}")
        print(f"your hand: {' '.join([card_to_str(c) for c in env.cards[human_idx]])}")
        print(f"board: {' '.join([card_to_str(c) for c in env.board])}")
        print(f"CHIPS: You {current_stacks[human_idx]:.2f} | Bot {current_stacks[bot_idx]:.2f}")
        input("\npress enter...")

if __name__ == "__main__":
    play()
