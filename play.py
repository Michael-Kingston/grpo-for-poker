import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time
import torch.nn.functional as F
from pokerbot.config import DEVICE, STATIC_DIM
from pokerbot.models import DynamicPokerLSTM
from pokerbot.poker_env import PokerState

# card mapping
CARD_MAP = {0: 'J', 1: 'Q', 2: 'K', None: 'None'}

def print_game_state(env, human_idx=1):
    print("\n" + "="*40)
    # pot/board
    board_str = CARD_MAP.get(env.comm_card, "Hidden")
    print(f"POT: {env.pot:.2f}  |  BOARD: {board_str}")
    print("-" * 40)
    
    # stacks
    bot_idx = 1 - human_idx
    print(f"BOT Stack: {env.stacks[bot_idx]:.2f}  |  In Front: {env.chips_in_front[bot_idx]:.2f}")
    print(f"YOU Stack: {env.stacks[human_idx]:.2f}  |  In Front: {env.chips_in_front[human_idx]:.2f}")
    
    # human card
    my_card = CARD_MAP[env.cards[human_idx]]
    print(f"YOUR HAND: [{my_card}]")
    print("="*40)

def get_human_action(env, p_idx):
    mask = env.get_mask(p_idx).cpu().numpy()
    # mask: [Fold, Call, Bet]
    
    print("\nOptions:")
    options = []
    if mask[0] == 1: options.append("f (Fold)")
    if mask[1] == 1: options.append("c (Call/Check)")
    if mask[2] == 1: options.append("b (Bet/Raise)")
    
    print(" | ".join(options))
    
    while True:
        choice = input("Your Move: ").strip().lower()
        
        # fold
        if choice == 'f' and mask[0] == 1:
            return 0, 0.0
            
        # call
        if choice == 'c' and mask[1] == 1:
            return 1, 0.0
            
        # bet
        if choice == 'b' and mask[2] == 1:
            # calc bet range
            opp_idx = 1 - p_idx
            curr_opp_bet = env.chips_in_front[opp_idx]
            my_current = env.chips_in_front[p_idx]
            min_raise_const = 2.0
            
            call_cost = max(0, curr_opp_bet - my_current)
            available = max(0, env.stacks[p_idx] - call_cost)
            
            min_bet_amt = min_raise_const
            max_bet_amt = available
            
            # handle tiny stack
            if min_bet_amt > max_bet_amt:
                min_bet_amt = max_bet_amt
            
            print(f"Bet Amount (Min {min_bet_amt:.2f} - Max {max_bet_amt:.2f}):")
            try:
                amt_input = float(input("> "))
            except ValueError:
                print("Invalid number.")
                continue
                
            if amt_input < min_bet_amt or amt_input > max_bet_amt:
                print("Amount out of range.")
                continue
                
            # frac conversion
            if max_bet_amt - min_bet_amt < 1e-9:
                frac = 1.0
            else:
                frac = (amt_input - min_bet_amt) / (max_bet_amt - min_bet_amt)
            return 2, frac
            
        print("Invalid choice.")

def play():
    print("Loading Model...")
    policy = DynamicPokerLSTM().to(DEVICE)
    try:
        policy.load_state_dict(torch.load("poker_model.pt", map_location=DEVICE))
        policy.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file 'poker_model.pt' not found. Run train.py first.")
        return

    env = PokerState()
    human_idx = 1
    bot_idx = 0
    
    # starting buy-in
    current_stacks = [100.0, 100.0]
    
    while True:
        # check game over
        if current_stacks[0] < 1.0 or current_stacks[1] < 1.0:
            print("\n!!! GAME OVER !!!")
            if current_stacks[human_idx] > current_stacks[bot_idx]:
                print("YOU CLEARED THE TABLE!")
            else:
                print("YOU WENT BUST!")
            break

        env.reset(starting_stacks=current_stacks)
        print(f"\n\n>>> NEW HAND <<<")
        
        curr = 0 # small blind
        curr_turn = 0
        
        while not env.finished:
            is_human_turn = (curr_turn == human_idx)
            
            obs = env.get_obs(curr_turn)
            mask = env.get_mask(curr_turn)
            
            if is_human_turn:
                print_game_state(env, human_idx)
                cat, amt_frac = get_human_action(env, human_idx)
                env.step(human_idx, cat, amt_frac)
                
                act_str = ["Fold", "Call", "Bet"][cat]
                print(f"You chose: {act_str}")
                
            else:
                # bot turn
                time.sleep(0.5) 
                
                with torch.no_grad():
                    cat, amt_frac, _ = policy.get_action(obs, mask)
                
                env.step(bot_idx, cat, amt_frac)
                
                act_str = ["Fold", "Call", "Bet"][cat]
                if cat == 2:
                    # just verify bet
                    opp_idx = 1 - bot_idx
                    print(f"\n> BOT chooses: {act_str}")
                else:
                    print(f"\n> BOT chooses: {act_str}")
            
            curr_turn = 1 - curr_turn
            
        # hand over
        print("\n--- ROUND OVER ---")
        if env.winner_idx == -1:
             env.resolve_showdown()
             
        # calc payout
        payoff_human = env.get_payoff(human_idx)
        
        # update internal env state
        final_human_stack = env.stacks[human_idx]
        final_bot_stack = env.stacks[bot_idx]
        
        if env.winner_idx == human_idx:
            final_human_stack += env.pot
        elif env.winner_idx == bot_idx:
            final_bot_stack += env.pot
        else:
            final_human_stack += (env.pot / 2.0)
            final_bot_stack += (env.pot / 2.0)
            
        # update persistent stacks
        current_stacks[human_idx] = final_human_stack
        current_stacks[bot_idx] = final_bot_stack
        
        if env.winner_idx == human_idx:
            print(f"WINNER: YOU (+{payoff_human:.2f})")
        elif env.winner_idx == bot_idx:
            print(f"WINNER: BOT (+{env.get_payoff(bot_idx):.2f})")
        else:
            print("SPLIT / DRAW")
            
        print(f"Bot Hand: {CARD_MAP[env.cards[bot_idx]]}")
        print(f"Your Hand: {CARD_MAP[env.cards[human_idx]]}")
        
        if env.comm_card:
            print(f"Board: {CARD_MAP[env.comm_card]}")
        
        print("-" * 20)
        print(f"CHIPS: You {current_stacks[human_idx]:.2f} | Bot {current_stacks[bot_idx]:.2f}")
        input("Press Enter for next hand...")

if __name__ == "__main__":
    play()
