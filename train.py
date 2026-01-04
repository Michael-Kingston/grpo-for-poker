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
    
    # until decision (sequential)
    while not env.finished:
        obs = env.get_obs(curr).to(DEVICE)
        mask = env.get_mask(curr).to(DEVICE)
        
        if curr == 0: 
            break
        else: 
            c_opp, a_opp = opp_pool.get_action(obs, mask, policy, env, curr)
            env.step(curr, c_opp, a_opp)
            curr = 1 - curr
            
    if env.finished:
        return [] 
        
    # multiverse (vectorized)
    snapshot = env.save_state()
    
    # initial sampling
    obs_b = snapshot.get_obs(0).to(DEVICE).unsqueeze(0)
    mask_b = snapshot.get_mask(0).to(DEVICE).unsqueeze(0)
    
    with torch.inference_mode():
        logits_b, alpha_b, beta_b = policy(obs_b, mask_b)
        
        # temp
        temp = 1.5
        logits_b = logits_b / max(temp, 1e-6)
        alpha_b = (alpha_b - 1.0) / max(temp, 1e-6) + 1.0
        beta_b = (beta_b - 1.0) / max(temp, 1e-6) + 1.0
        
        # dists
        cat_dist = dist.Categorical(F.softmax(logits_b, dim=-1))
        amt_dist = dist.Beta(alpha_b, beta_b)
        
        # sampling
        cat_actions = cat_dist.sample((GROUP_SIZE,)).squeeze()
        amt_actions = amt_dist.sample((GROUP_SIZE,)).squeeze()
        
        # lps
        lp_cats = cat_dist.log_prob(cat_actions).squeeze()
        lp_amts = amt_dist.log_prob(amt_actions).squeeze()
        is_bet_b = (cat_actions == 2).float()
        total_lps = lp_cats + (lp_amts * is_bet_b)
        
    # setup environments
    sim_envs = [snapshot.save_state() for _ in range(GROUP_SIZE)]
    for i, s_env in enumerate(sim_envs):
        s_env.step(0, cat_actions[i].item(), amt_actions[i].item())
        
    # parallel rollout
    p_turn = 1 # next is opp
    while any(not e.finished for e in sim_envs):
        active_indices = [i for i, e in enumerate(sim_envs) if not e.finished]
        if not active_indices: break
        
        active_envs = [sim_envs[i] for i in active_indices]
        # batch transfer: stack on CPU, move ONCE to DEVICE
        obs_batch = torch.stack([e.get_obs(p_turn) for e in active_envs]).to(DEVICE)
        mask_batch = torch.stack([e.get_mask(p_turn) for e in active_envs]).to(DEVICE)
        
        next_actions = [] 
        
        if p_turn == 1: # opp
            # opp pool logic - batch the self-play ones
            next_actions_map = {} # env_idx -> (cat, amt, none)
            
            # 1. identify who needs what
            needs_self = [] 
            needs_other = [] # (env_idx, action_type)
            
            for i, env_idx in enumerate(active_indices):
                r = random.random()
                if opp_pool.mode == "bootcamp":
                    if r < 0.2: needs_self.append(i)
                    else: needs_other.append((i, "fish"))
                else: # normal
                    if r < 0.1: needs_other.append((i, "fish"))
                    elif r < 0.35 and len(opp_pool.historical) > 0:
                        needs_other.append((i, "ghost"))
                    else: needs_self.append(i)
            
            # 2. handle others sequentially (small subset)
            for i, o_type in needs_other:
                env_idx = active_indices[i]
                if o_type == "fish":
                    mask = mask_batch[i]
                    if mask[1] == 1: c, a = 1, 0.0
                    elif mask[0] == 1: c, a = 0, 0.0
                    else: c, a = 1, 0.0
                else: # ghost
                    opp_policy = random.choice(opp_pool.historical)
                    with torch.no_grad():
                        c, a, _ = opp_policy.get_action(obs_batch[i], mask_batch[i])
                next_actions_map[i] = (c, a, None)
                
            # 3. handle self-play in batch
            if needs_self:
                self_obs = obs_batch[needs_self]
                self_mask = mask_batch[needs_self]
                with torch.inference_mode():
                    c_logits, alpha, beta = policy(self_obs, self_mask)
                    c_dist = dist.Categorical(F.softmax(c_logits, dim=-1))
                    c_act = c_dist.sample()
                    a_dist = dist.Beta(alpha, beta)
                    a_act = a_dist.sample()
                    
                    for j, i in enumerate(needs_self):
                        next_actions_map[i] = (c_act[j].item(), a_act[j].item(), None)
            
            # collate in original order
            for i in range(len(active_indices)):
                next_actions.append(next_actions_map[i])
                
        else: # hero
            with torch.inference_mode():
                # batched hero call
                c_logits, alpha, beta = policy(obs_batch, mask_batch)
                
                # temp 1.0
                c_probs = F.softmax(c_logits, dim=-1)
                c_dist = dist.Categorical(c_probs)
                c_act = c_dist.sample()
                
                a_dist = dist.Beta(alpha, beta)
                a_act = a_dist.sample()
                
                for j in range(len(active_indices)):
                    next_actions.append((c_act[j].item(), a_act[j].item(), None))
                    
        # apply
        for j, idx in enumerate(active_indices):
            c_a, a_a, _ = next_actions[j]
            sim_envs[idx].step(p_turn, c_a, a_a)
            
        p_turn = 1 - p_turn

    # results
    group_results = []
    for i in range(GROUP_SIZE):
        reward = sim_envs[i].get_payoff(0)
        # obs/mask here should be on DEVICE 
        # (originally hero decision point obs, which we have as obs_b)
        group_results.append({
            'obs': obs_b.squeeze(0), 'mask': mask_b.squeeze(0), 'cat': cat_actions[i].item(), 'amt': amt_actions[i].item(), 'lp': total_lps[i], 'reward': reward
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
