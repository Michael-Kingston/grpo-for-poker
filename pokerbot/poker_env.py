import random
import torch
from .config import DEVICE, MAX_HISTORY, STATIC_DIM, TOTAL_OBS_DIM, START_STACK, MIN_RAISE
from .evaluator import evaluate_7_cards

class PokerState:
    def __init__(self):
        self.start_stack = START_STACK
        self.cards = [[], []] # hole cards
        self.board = []
        self.deck = []
        
        self.stacks = [0.0, 0.0]
        self.chips_in_front = [0.0, 0.0]
        self.pot = 0.0
        
        self.round = 1 
        self.finished = False
        self.winner_idx = -1
        self.history = [] 
        self.actions_this_round = 0

    def reset(self, rigged_deck=None, starting_stacks=None):
        if rigged_deck is not None:
            self.deck = rigged_deck
        else:
            self.deck = list(range(52))
            random.shuffle(self.deck)
        
        # hole cards
        self.cards[0] = [self.deck.pop(0), self.deck.pop(0)]
        self.cards[1] = [self.deck.pop(0), self.deck.pop(0)]
        self.board = []
        
        if starting_stacks is not None:
            self.stacks = list(starting_stacks)
        else:
            self.stacks = [self.start_stack, self.start_stack]
            
        self.chips_in_front = [0.0, 0.0] 
        self.pot = 0.0
        
        self.post_blind(0, 1.0)
        self.post_blind(1, 2.0)
        
        self.round = 1 # preflop
        self.finished = False
        self.winner_idx = -1
        self.history = [] 
        self.actions_this_round = 0

    def post_blind(self, p_idx, amt):
        self.stacks[p_idx] -= amt
        self.chips_in_front[p_idx] += amt
        self.pot += amt

    def save_state(self):
        # manual copy faster
        new_state = PokerState()
        new_state.start_stack = self.start_stack
        new_state.cards = [list(c) for c in self.cards]
        new_state.board = list(self.board)
        new_state.deck = list(self.deck)
        new_state.stacks = list(self.stacks)
        new_state.chips_in_front = list(self.chips_in_front)
        new_state.pot = self.pot
        new_state.round = self.round
        new_state.finished = self.finished
        new_state.winner_idx = self.winner_idx
        new_state.history = list(self.history)
        new_state.actions_this_round = self.actions_this_round
        return new_state

    def load_state(self, snapshot):
        self.__dict__.update(snapshot.__dict__)

    def get_obs(self, p_idx):
        # cpu fill faster
        obs = torch.zeros(TOTAL_OBS_DIM)
        
        # hero cards
        my_cards = self.cards[p_idx]
        for i, c in enumerate(my_cards):
            rank, suit = c // 4, c % 4
            obs[i * 17 + rank] = 1.0
            obs[i * 17 + 13 + suit] = 1.0
            
        # board cards
        for i in range(5):
            idx = 34 + (i * 18)
            if i < len(self.board):
                c = self.board[i]
                rank, suit = c // 4, c % 4
                obs[idx + rank] = 1.0
                obs[idx + 13 + suit] = 1.0
                obs[idx + 17] = 1.0 # visible
                
        # numeric features
        obs[124] = self.stacks[p_idx] / self.start_stack
        obs[125] = self.stacks[1-p_idx] / self.start_stack
        obs[126] = self.chips_in_front[p_idx] / self.start_stack
        obs[127] = self.chips_in_front[1-p_idx] / self.start_stack
        obs[128] = self.pot / (self.start_stack * 2)
        obs[129] = self.round / 4.0
        
        opp_idx = 1 - p_idx
        to_call = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
        pot_total = self.pot + to_call
        
        # pot odds
        if pot_total > 0: 
            obs[130] = to_call / pot_total

        # spr
        if self.pot > 0:
            obs[131] = self.stacks[p_idx] / self.pot
        else:
            obs[131] = self.stacks[p_idx]
            
        # mask
        mask = self.get_mask(p_idx).cpu()
        obs[132:135] = mask

        # history
        if len(self.history) > 0:
            relevant = self.history[-MAX_HISTORY:]
            start = STATIC_DIM + (MAX_HISTORY - len(relevant))
            for i, (cat, amt) in enumerate(relevant):
                val = 0.33 if cat == 0 else (0.66 if cat == 1 else 1.0)
                obs[start + i] = val
        return obs

    def get_mask(self, p_idx):
        mask = torch.ones(3)
        opp_idx = 1 - p_idx
        to_call = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
        if to_call <= 0.01: mask[0] = 0 # no fold on check
        if self.stacks[p_idx] <= 0.01: mask[2] = 0 # no bet if allin
        return mask

    def step(self, p_idx, category, amount_frac):
        self.history.append((category, amount_frac))
        self.actions_this_round += 1
        opp_idx = 1 - p_idx
        
        if category == 0: # fold
            self.finished = True
            self.winner_idx = opp_idx
            return

        if category == 1: # call
            diff = self.chips_in_front[opp_idx] - self.chips_in_front[p_idx]
            call_amt = min(diff, self.stacks[p_idx])
            self.stacks[p_idx] -= call_amt
            self.chips_in_front[p_idx] += call_amt
            self.pot += call_amt
            
            # round progression
            if self.chips_in_front[0] == self.chips_in_front[1] and self.actions_this_round >= 2:
                if self.round < 4:
                    self.round += 1
                    self.chips_in_front = [0.0, 0.0]
                    self.actions_this_round = 0
                    if self.round == 2: # flop
                        self.board += [self.deck.pop(0), self.deck.pop(0), self.deck.pop(0)]
                    elif self.round == 3 or self.round == 4: # turn/river
                        self.board.append(self.deck.pop(0))
                else:
                    self.finished = True
                    self.resolve_showdown()

        if category == 2: # bet
            curr_opp_bet = self.chips_in_front[opp_idx]
            my_current = self.chips_in_front[p_idx]
            min_raise = MIN_RAISE
            
            call_cost = max(0, curr_opp_bet - my_current)
            available_stack = max(0, self.stacks[p_idx] - call_cost)
            
            if available_stack < 0.01:
                self.step(p_idx, 1, 0.0) # auto call
                return

            raise_amt = min_raise + (amount_frac * (available_stack - min_raise))
            raise_amt = min(raise_amt, available_stack)
            total_cost = call_cost + raise_amt
            
            self.stacks[p_idx] -= total_cost
            self.chips_in_front[p_idx] += total_cost
            self.pot += total_cost

    def resolve_showdown(self):
        score0 = evaluate_7_cards(self.cards[0] + self.board)
        score1 = evaluate_7_cards(self.cards[1] + self.board)
        
        if score0 > score1: self.winner_idx = 0
        elif score1 > score0: self.winner_idx = 1
        else: self.winner_idx = -1

    def get_payoff(self, p_idx):
        current = self.stacks[p_idx]
        if self.winner_idx == p_idx: current += self.pot
        elif self.winner_idx == -1: current += (self.pot / 2.0)
        return current - self.start_stack
