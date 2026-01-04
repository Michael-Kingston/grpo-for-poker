import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
from collections import deque
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from pokerbot.config import DEVICE, BATCH_SIZE, GROUP_SIZE, PPO_EPOCHS, CLIP_EPS, ENTROPY_COEFF, LR
from pokerbot.utils import StepData, set_seed
from pokerbot.poker_env import PokerState
from pokerbot.models import DynamicPokerLSTM

class OpponentPool:
    def __init__(self):
        self.historical = deque(maxlen=3)
        self.mode = "bootcamp" 
        
    def snapshot(self, policy):
        self.historical.append(copy.deepcopy(policy))
        
    def get_action(self, obs, mask, policy, env, p_idx):
        # bootcamp
        if self.mode == "bootcamp":
            # 20% self-play mix
            if random.random() < 0.2:
                with torch.no_grad():
                    c, a, _ = policy.get_action(obs, mask)
                return c, a
            
            # 80% fish
            if mask[1] == 1:
                return 1, 0.0 
            elif mask[0] == 1:
                return 0, 0.0 
            return 1, 0.0
            
        # normal
        else:
            r = random.random()
            # weak (10%)
            if r < 0.1:
                if mask[1] == 1: return 1, 0.0
                return 0, 0.0
            
            # historic (25%)
            if r < 0.35 and len(self.historical) > 0:
                opp_policy = random.choice(self.historical)
                with torch.no_grad():
                    c, a, _ = opp_policy.get_action(obs, mask)
                return c, a
                
            # current self (65%)
            with torch.no_grad():
                c, a, _ = policy.get_action(obs, mask)
            return c, a

def collect_group_trajectories(policy, env, opp_pool):
    # plays until hero decision then multiverse
    buffer = []
    
    env.reset()
    curr = 0 
    
    # until decision
    while not env.finished:
        obs = env.get_obs(curr)
        mask = env.get_mask(curr)
        
        if curr == 0: 
            break
        else: 
            c_opp, a_opp = opp_pool.get_action(obs, mask, policy, env, curr)
            env.step(curr, c_opp, a_opp)
            curr = 1 - curr
            
    if env.finished:
        return [] 
        
    # multiverse
    snapshot = env.save_state()
    group_results = []
    
    obs_b = snapshot.get_obs(0).unsqueeze(0)
    mask_b = snapshot.get_mask(0).unsqueeze(0)
    
    # inference mode faster
    with torch.inference_mode():
        logits_b, alpha_b, beta_b = policy(obs_b, mask_b)
        
        # temp
        logits_b = logits_b / max(1.5, 1e-6)
        alpha_b = (alpha_b - 1.0) / max(1.5, 1e-6) + 1.0
        beta_b = (beta_b - 1.0) / max(1.5, 1e-6) + 1.0
        
        # dists
        cat_dist = dist.Categorical(F.softmax(logits_b, dim=-1))
        amt_dist = dist.Beta(alpha_b, beta_b)
        
        # batched sampling faster
        cat_actions = cat_dist.sample((GROUP_SIZE,)).squeeze()
        amt_actions = amt_dist.sample((GROUP_SIZE,)).squeeze()
        
        # lps
        lp_cats = cat_dist.log_prob(cat_actions).squeeze()
        lp_amts = amt_dist.log_prob(amt_actions).squeeze()
        is_bet_b = (cat_actions == 2).float()
        total_lps = lp_cats + (lp_amts * is_bet_b)
        
    for i in range(GROUP_SIZE):
        sim_env = snapshot.save_state()
        
        c_act = cat_actions[i].item()
        a_act = amt_actions[i].item()
        lp = total_lps[i]
        
        sim_env.step(0, c_act, a_act)
        
        p_turn = 1
        while not sim_env.finished:
            s_obs = sim_env.get_obs(p_turn)
            s_mask = sim_env.get_mask(p_turn)
            
            if p_turn == 1: # opp
                c_o, a_o = opp_pool.get_action(s_obs, s_mask, policy, sim_env, p_turn)
                sim_env.step(p_turn, c_o, a_o)
            else: # hero
                with torch.inference_mode():
                    c_o, a_o, _ = policy.get_action(s_obs, s_mask, temperature=1.0)
                sim_env.step(p_turn, c_o, a_o)
            p_turn = 1 - p_turn
            
        reward = sim_env.get_payoff(0)
        group_results.append({
            'obs': obs, 'mask': mask, 'cat': c_act, 'amt': a_act, 'lp': lp, 'reward': reward
        })
        
    # advantages
    rewards = torch.tensor([r['reward'] for r in group_results], device=DEVICE)
    mean = rewards.mean()
    std = rewards.std() + 1e-6
    advantages = (rewards - mean) / std
    
    for i, res in enumerate(group_results):
        buffer.append(StepData(res['obs'], res['mask'], res['cat'], res['amt'], res['lp'], advantages[i].item()))
        
    return buffer

def train():
    set_seed(42)
    policy = DynamicPokerLSTM().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    env = PokerState()
    opp_pool = OpponentPool()
    
    print(f"starting training on {DEVICE}")
    
    for i in range(1, 10001):
        if i >= 201 and opp_pool.mode == "bootcamp":
            opp_pool.mode = "normal"
            print(">>> graduated to normal mode <<<")
            
        if i % 50 == 0:
            opp_pool.snapshot(policy)
            
        buffer_obs, buffer_mask, buffer_cat, buffer_amt, buffer_lp, buffer_adv = [], [], [], [], [], []
        
        # batching
        while len(buffer_obs) < BATCH_SIZE:
            trajs = collect_group_trajectories(policy, env, opp_pool)
            for step in trajs:
                buffer_obs.append(step.obs)
                buffer_mask.append(step.mask)
                buffer_cat.append(step.cat_action)
                buffer_amt.append(step.amt_action)
                buffer_lp.append(step.log_prob)
                buffer_adv.append(step.reward) 
        
        t_obs = torch.stack(buffer_obs)
        t_mask = torch.stack(buffer_mask)
        t_cat = torch.tensor(buffer_cat, device=DEVICE)
        t_amt = torch.tensor(buffer_amt, device=DEVICE).unsqueeze(-1)
        t_old_lp = torch.stack(buffer_lp).detach()
        t_adv = torch.tensor(buffer_adv, device=DEVICE).detach()
        
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
            
            print(f"iter {i} | loss: {loss.item():.4f} | adv: {t_adv.mean():.4f}")
            print(f"stats: fold {n_fold/total_acts:.2f} | call {n_call/total_acts:.2f} | bet {n_bet/total_acts:.2f}")
            print(f"mode: {opp_pool.mode}")

        if i % 100 == 0:
            torch.save(policy.state_dict(), "poker_model.pt")
            print(f"checkpoint saved at iter {i}")
            
    torch.save(policy.state_dict(), "poker_model.pt")
    print("model saved")

if __name__ == "__main__":
    train()
