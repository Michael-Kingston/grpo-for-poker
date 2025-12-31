import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
from collections import deque
import random
import torch
import torch.optim as optim
import torch.distributions as dist
from pokerbot.config import DEVICE, BATCH_SIZE, GROUP_SIZE, PPO_EPOCHS, CLIP_EPS, ENTROPY_COEFF, LR
from pokerbot.utils import StepData, set_seed
from pokerbot.poker_env import PokerState
from pokerbot.models import DynamicPokerLSTM

class OpponentPool:
    def __init__(self):
        self.historical = deque(maxlen=3)
        self.mode = "bootcamp" # start easy
        
    def snapshot(self, policy):
        self.historical.append(copy.deepcopy(policy))
        
    def get_action(self, obs, mask, policy, env, p_idx):
        # bootcamp calling station
        if self.mode == "bootcamp":
            # check call usually fold 5pct
            if mask[1] == 1:
                return 1, 0.0 # call
            elif mask[0] == 1:
                return 0, 0.0 # fold
            return 1, 0.0
            
        # normal mixed strategy
        else:
            r = random.random()
            # 20% weak (call/check)
            if r < 0.2:
                if mask[1] == 1: return 1, 0.0
                return 0, 0.0
            
            # 20pct historic self
            if r < 0.4 and len(self.historical) > 0:
                opp_policy = random.choice(self.historical)
                with torch.no_grad():
                    c, a, _ = opp_policy.get_action(obs, mask)
                return c, a
                
            # 60pct current self nash
            with torch.no_grad():
                c, a, _ = policy.get_action(obs, mask)
            return c, a

def collect_group_trajectories(policy, env, opp_pool):
    trajectories = []
    n_bets = 0
    n_actions = 0
    
    for _ in range(GROUP_SIZE):
        env.reset()
        traj = []
        curr = 0 # sb acts first preflop
        
        while not env.finished:
            obs = env.get_obs(curr)
            mask = env.get_mask(curr)
            
            if curr == 1: # opp
                c_opp, a_opp = opp_pool.get_action(obs, mask, policy, env, curr)
                env.step(curr, c_opp, a_opp)
            else: # hero
                with torch.no_grad():
                    c_act, a_act, lp = policy.get_action(obs, mask)
                
                n_actions += 1
                if c_act == 2: n_bets += 1
                
                traj.append(StepData(obs, mask, c_act, a_act, lp, 0))
                env.step(0, c_act, a_act)
            
            curr = 1 - curr
            
        payoff = env.get_payoff(0)
        if traj:
            last = traj[-1]
            traj[-1] = StepData(last.obs, last.mask, last.cat_action, last.amt_action, last.log_prob, payoff)
        trajectories.append(traj)
    return trajectories, n_bets, n_actions

def train():
    set_seed(42)
    policy = DynamicPokerLSTM().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = PokerState()
    opp_pool = OpponentPool()
    
    print(f"starting training on {DEVICE}")
    print("[iteration 2] nlth 1v1 | 52-card deck | 4 rounds")
    
    for i in range(1, 10001):
        if i % 50 == 0:
            opp_pool.snapshot(policy)
            
        buffer_obs, buffer_mask, buffer_cat, buffer_amt, buffer_lp, buffer_adv = [], [], [], [], [], []
        total_bets, total_actions = 0, 0
        
        # collect groups for batch size
        for _ in range(3):
            trajs, nb, na = collect_group_trajectories(policy, env, opp_pool)
            total_bets += nb
            total_actions += na
            
            rewards = torch.tensor([sum(s.reward for s in t) for t in trajs], device=DEVICE)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            
            for j, t in enumerate(trajs):
                for step in t:
                    buffer_obs.append(step.obs)
                    buffer_mask.append(step.mask)
                    buffer_cat.append(step.cat_action)
                    buffer_amt.append(step.amt_action)
                    buffer_lp.append(step.log_prob)
                    buffer_adv.append(adv[j])
        
        # adaptive logic
        bet_ratio = total_bets / (total_actions + 1e-9)
        if bet_ratio > 0.25: # aggressive to normal
            opp_pool.mode = "normal"
        else:
            opp_pool.mode = "bootcamp" # passive to bootcamp
            
        if len(buffer_obs) < BATCH_SIZE: 
            continue
        
        t_obs = torch.stack(buffer_obs)
        t_mask = torch.stack(buffer_mask)
        t_cat = torch.tensor(buffer_cat, device=DEVICE)
        t_amt = torch.tensor(buffer_amt, device=DEVICE).unsqueeze(-1)
        t_old_lp = torch.stack(buffer_lp).detach()
        t_adv = torch.stack(buffer_adv).detach()
        
        for _ in range(PPO_EPOCHS):
            cat_logits, alpha, beta = policy(t_obs, t_mask)
            
            cat_dist = dist.Categorical(logits=cat_logits)
            new_cat_lp = cat_dist.log_prob(t_cat)
            
            amt_dist = dist.Beta(alpha, beta)
            new_amt_lp = amt_dist.log_prob(t_amt).squeeze(-1)
            
            is_bet = (t_cat == 2).float()
            new_lp = new_cat_lp + (new_amt_lp * is_bet)
            
            ratio = torch.exp(new_lp - t_old_lp)
            surr1 = ratio * t_adv
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * t_adv
            loss = -torch.min(surr1, surr2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if i % 10 == 0:
            # stats
            total_acts = len(t_cat)
            n_fold = (t_cat == 0).sum().item()
            n_call = (t_cat == 1).sum().item()
            n_bet = (t_cat == 2).sum().item()
            
            print(f"Iter {i} | Loss: {loss.item():.4f} | Adv: {t_adv.mean():.4f}")
            print(f"Stats: Fold {n_fold/total_acts:.2f} | Call {n_call/total_acts:.2f} | Bet {n_bet/total_acts:.2f}")
            print(f"Mode: {opp_pool.mode} | Bet Ratio: {bet_ratio:.2f}")

        if i % 100 == 0:
            torch.save(policy.state_dict(), "poker_model.pt")
            print(f"checkpoint saved at iter {i}")
            
    torch.save(policy.state_dict(), "poker_model.pt")
    print("model saved")

if __name__ == "__main__":
    train()
