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
        
    def pick_strategy(self):
        # returns (type, policy_obj_or_none)
        r = random.random()
        if self.mode == "bootcamp":
            if r < 0.2: return "self", None
            return "fish", None
        else: # normal
            if r < 0.1: return "fish", None
            if r < 0.35 and self.historical:
                return "ghost", random.choice(self.historical)
            return "self", None

    def get_strategy_action(self, strategy, policy, obs, mask):
        # strategy is (type, ghost_obj)
        s_type, ghost = strategy
        if s_type == "self":
            with torch.no_grad():
                c, a, _ = policy.get_action(obs, mask)
            return c, a
        elif s_type == "ghost":
            with torch.no_grad():
                c, a, _ = ghost.get_action(obs, mask)
            return c, a
        else: # fish
            if mask[1] == 1: return 1, 0.0
            if mask[0] == 1: return 0, 0.0
            return 1, 0.0

def collect_group_trajectories(policy, env, opp_pool):
    # plays until hero decision then multiverse
    buffer = []
    env.reset()
    curr = 0 
    
    # lock in opponent for this hand
    strategy = opp_pool.pick_strategy()
    
    # sequential until hero decision
    while not env.finished:
        if curr == 0: break
        obs = env.get_obs(curr).to(DEVICE)
        mask = env.get_mask(curr).to(DEVICE)
        c, a = opp_pool.get_strategy_action(strategy, policy, obs, mask)
        env.step(curr, c, a)
        curr = 1 - curr
            
    if env.finished: return [] 
        
    # multiverse (vectorised)
    snapshot = env.save_state()
    obs_b = snapshot.get_obs(0).to(DEVICE)
    mask_b = snapshot.get_mask(0).to(DEVICE)
    
    with torch.inference_mode():
        logits_b, alpha_b, beta_b = policy(obs_b.unsqueeze(0), mask_b.unsqueeze(0))
        # fully squeeze to scalars for the distributions
        l_b = logits_b.squeeze(0)
        a_b = alpha_b.view(-1)[0]
        b_b = beta_b.view(-1)[0]
        
        cat_dist = dist.Categorical(logits=l_b)
        amt_dist = dist.Beta(a_b, b_b)
        
        # sample initial multiverse actions
        cat_actions = cat_dist.sample((GROUP_SIZE,))
        amt_actions = amt_dist.sample((GROUP_SIZE,))
        
        # log probs
        lp_cats = cat_dist.log_prob(cat_actions)
        lp_amts = amt_dist.log_prob(amt_actions)
        is_bet_b = (cat_actions == 2).float()
        total_lps = lp_cats + (lp_amts * is_bet_b)
        
    # setup environments
    sim_envs = [snapshot.save_state() for _ in range(GROUP_SIZE)]
    for i in range(GROUP_SIZE):
        sim_envs[i].step(0, cat_actions[i].item(), amt_actions[i].item())
        
    # parallel rollout
    p_turn = 1 
    s_type, ghost = strategy
    
    while True:
        # get active envs
        active = [e for e in sim_envs if not e.finished]
        if not active: break
        
        num_active = len(active)
        # batch transfer
        obs_batch = torch.stack([e.get_obs(p_turn) for e in active]).to(DEVICE)
        mask_batch = torch.stack([e.get_mask(p_turn) for e in active]).to(DEVICE)
        
        if p_turn == 1: # opp
            if s_type == "fish":
                for env_idx in range(num_active):
                    m = mask_batch[env_idx]
                    c = 1 if m[1] == 1 else (0 if m[0] == 1 else 1)
                    active[env_idx].step(p_turn, c, 0.0)
            elif s_type == "ghost":
                with torch.no_grad():
                    c_l, alpha, beta = ghost(obs_batch, mask_batch)
                    c_act = dist.Categorical(logits=c_l).sample()
                    a_act = dist.Beta(alpha, beta).sample()
                    for j in range(num_active):
                        active[j].step(p_turn, c_act[j].item(), a_act[j].item())
            else: # self
                with torch.inference_mode():
                    c_l, alpha, beta = policy(obs_batch, mask_batch)
                    c_act = dist.Categorical(logits=c_l).sample()
                    a_act = dist.Beta(alpha, beta).sample()
                    for j in range(num_active):
                        active[j].step(p_turn, c_act[j].item(), a_act[j].item())
        else: # hero
            with torch.inference_mode():
                c_l, alpha, beta = policy(obs_batch, mask_batch)
                c_act = dist.Categorical(logits=c_l).sample()
                a_act = dist.Beta(alpha, beta).sample()
                for j in range(num_active):
                    active[j].step(p_turn, c_act[j].item(), a_act[j].item())
                    
        p_turn = 1 - p_turn

    # results
    group_results_reward = [e.get_payoff(0) for e in sim_envs]
    rewards = torch.tensor(group_results_reward, device=DEVICE)
    mean = rewards.mean()
    std = rewards.std() + 1e-6
    advantages = (rewards - mean) / std
    adv_cpu = advantages.cpu()
    
    # hero decision point tensors
    o_fixed = obs_b.squeeze(0)
    m_fixed = mask_b.squeeze(0)
    
    for i in range(GROUP_SIZE):
        buffer.append(StepData(o_fixed, m_fixed, cat_actions[i].item(), amt_actions[i].item(), total_lps[i], adv_cpu[i].item()))
        
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
            
        # pre-allocated buffers for speed
        b_obs = []
        b_mask = []
        b_cat = []
        b_amt = []
        b_lp = []
        b_adv = []
        
        # collection
        while len(b_obs) < BATCH_SIZE:
            trajs = collect_group_trajectories(policy, env, opp_pool)
            for step in trajs:
                b_obs.append(step.obs)
                b_mask.append(step.mask)
                b_cat.append(step.cat_action)
                b_amt.append(step.amt_action)
                b_lp.append(step.log_prob)
                b_adv.append(step.reward) # grpo stores advantage in reward field
        
        # batch preparation
        t_obs = torch.stack(b_obs[:BATCH_SIZE])
        t_mask = torch.stack(b_mask[:BATCH_SIZE])
        t_cat = torch.tensor(b_cat[:BATCH_SIZE], device=DEVICE)
        t_amt = torch.tensor(b_amt[:BATCH_SIZE], device=DEVICE).unsqueeze(-1)
        t_old_lp = torch.stack(b_lp[:BATCH_SIZE]).detach()
        t_adv = torch.tensor(b_adv[:BATCH_SIZE], device=DEVICE).detach()
        
        # ppo update 
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
