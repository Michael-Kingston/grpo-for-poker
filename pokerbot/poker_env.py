import random
import torch
from .config import DEVICE, MAX_HISTORY, STATIC_DIM, TOTAL_OBS_DIM, START_STACK, MIN_RAISE

class PokerState:
    def __init__(self):
        self.start_stack = START_STACK
        self.cards = [None, None] 
        self.comm_card = None
        self.deck_state = None
        
        self.stacks = [0.0, 0.0]
        self.chips_in_front = [0.0, 0.0]
        self.pot = 0.0
        
        self.round = 1 
        self.finished = False
        self.winner_idx = -1
        self.history = [] 

    def reset(self, rigged_deck=None, starting_stacks=None):
        if rigged_deck is not None:
            self.deck_state = rigged_deck
        else:
            self.deck_state = [0, 0, 1, 1, 2, 2]
            random.shuffle(self.deck_state)
            
        self.cards[0] = self.deck_state[0]
        self.cards[1] = self.deck_state[1]
        self.flop_card = self.deck_state[2] 
        self.comm_card = None
        
        if starting_stacks is not None:
            self.stacks = list(starting_stacks) # Copy values
        else:
            self.stacks = [self.start_stack, self.start_stack]
            
        self.chips_in_front = [0.0, 0.0] 
        self.pot = 0.0
        
        self.post_blind(0, 1.0)
        self.post_blind(1, 1.0)
        
        self.round = 1 
        self.finished = False
        self.winner_idx = -1
        self.history = [] 

    def post_blind(self, p_idx, amt):
        self.stacks[p_idx] -= amt
        self.chips_in_front[p_idx] += amt
        self.pot += amt

    def get_obs(self, p_idx):
        obs = torch.zeros(TOTAL_OBS_DIM, device=DEVICE)
        
        # static features
        obs[self.cards[p_idx]] = 1.0
        if self.comm_card is None: 
            obs[6] = 1.0
        else: 
            obs[3 + self.comm_card] = 1.0
        obs[7] = self.stacks[p_idx] / self.start_stack
        
        opp_idx = 1 - p_idx
        to_call = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
        pot_total = self.pot + to_call
        if pot_total > 0: 
            obs[8] = to_call / pot_total

        # dynamic history
        if len(self.history) > 0:
            relevant = self.history[-MAX_HISTORY:]
            start = STATIC_DIM + (MAX_HISTORY - len(relevant))
            for i, (cat, amt) in enumerate(relevant):
                # encode: fold=0, call=0.5, bet=1.0 + amount/2
                val = 0.0
                if cat == 1: val = 0.5
                if cat == 2: val = 0.6 + (amt * 0.4) 
                obs[start + i] = val
        return obs

    def get_mask(self, p_idx):
        # [fold, call, bet]
        mask = torch.ones(3, device=DEVICE)
        opp_idx = 1 - p_idx
        to_call = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
        
        # fold illegal if check is free
        if to_call <= 0: 
            mask[0] = 0
        
        # bet illegal if stack empty or < min_raise
        if self.stacks[p_idx] <= 0.01:
            mask[2] = 0 
        
        return mask

    def step(self, p_idx, category, amount_frac):
        self.history.append((category, amount_frac))
        opp_idx = 1 - p_idx
        
        # fold
        if category == 0:
            self.finished = True
            self.winner_idx = opp_idx
            return

        # call/check
        if category == 1:
            diff = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
            call_amt = min(diff, self.stacks[p_idx])
            self.stacks[p_idx] -= call_amt
            self.chips_in_front[p_idx] += call_amt
            self.pot += call_amt
            
            if self.chips_in_front[0] == self.chips_in_front[1]:
                if self.round == 1:
                    self.round = 2
                    self.chips_in_front = [0.0, 0.0]
                    self.comm_card = self.flop_card 
                    return
                else:
                    self.finished = True
                    self.resolve_showdown()
                    return

        # bet
        if category == 2:
            curr_opp_bet = self.chips_in_front[opp_idx]
            my_current = self.chips_in_front[p_idx]
            min_raise = MIN_RAISE
            
            call_cost = max(0, curr_opp_bet - my_current)
            available_stack = max(0, self.stacks[p_idx] - call_cost)
            
            if available_stack < 0.01:
                # forced call if math breaks
                self.step(p_idx, 1, 0.0) 
                return

            # map 0.0-1.0 to [minraise, allin]
            raise_amt = min_raise + (amount_frac * (available_stack - min_raise))
            raise_amt = min(raise_amt, available_stack)
            
            total_cost = call_cost + raise_amt
            
            self.stacks[p_idx] -= total_cost
            self.chips_in_front[p_idx] += total_cost
            self.pot += total_cost
            return

    def resolve_showdown(self):
        r1 = self.cards[0]; r2 = self.cards[1]; comm = self.comm_card
        s1 = (r1 == comm)*10 + r1
        s2 = (r2 == comm)*10 + r2
        if s1 > s2: self.winner_idx = 0
        elif s2 > s1: self.winner_idx = 1
        else: self.winner_idx = -1

    def get_payoff(self, p_idx):
        # Calculate Final Stack based on winner
        current_stack = self.stacks[p_idx]
        
        # If I won, I get the pot
        if self.winner_idx == p_idx:
            current_stack += self.pot
            
        # If Split, I get half
        elif self.winner_idx == -1:
            current_stack += (self.pot / 2.0)
            
        # If I lost, I get nothing (stack stays depleted)
        
        return current_stack - self.start_stack
