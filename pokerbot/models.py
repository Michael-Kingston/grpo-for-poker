import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from .config import STATIC_DIM

class DynamicPokerLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.static_net = nn.Sequential(nn.Linear(STATIC_DIM, 128), nn.ReLU())
        self.lstm = nn.LSTM(1, 128, batch_first=True)
        self.shared_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        
        # discrete category
        self.category_head = nn.Linear(128, 3)
        
        # continuous beta parameters
        self.amount_alpha = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.amount_beta =  nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        
    def forward(self, obs, mask=None):
        self.lstm.flatten_parameters()
        static = self.static_net(obs[:, :STATIC_DIM])
        
        hist = obs[:, STATIC_DIM:].unsqueeze(-1)
        _, (h_n, _) = self.lstm(hist)
        hist_feat = h_n[-1]
        
        combined = torch.cat([static, hist_feat], dim=1)
        latent = self.shared_fc(combined)
        
        cat_logits = self.category_head(latent)
        if mask is not None:
            cat_logits = cat_logits.masked_fill(mask == 0, -1e9)
            
        # valid beta shapes
        alpha = self.amount_alpha(latent) + 1.0
        beta = self.amount_beta(latent) + 1.0
        
        return cat_logits, alpha, beta
        
    def get_action(self, obs, mask):
        if obs.dim() == 1: 
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        cat_logits, alpha, beta = self.forward(obs, mask)
        
        # sample category
        cat_probs = F.softmax(cat_logits, dim=-1)
        cat_dist = dist.Categorical(cat_probs)
        cat_action = cat_dist.sample()
        
        # sample amount
        amt_dist = dist.Beta(alpha, beta)
        amt_action = amt_dist.sample()
        
        # stability clamp
        amt_action = torch.clamp(amt_action, 1e-6, 1.0 - 1e-6)
        
        # joint log prob
        log_prob_cat = cat_dist.log_prob(cat_action)
        log_prob_amt = amt_dist.log_prob(amt_action).squeeze(-1)
        
        is_bet = (cat_action == 2).float()
        total_log_prob = log_prob_cat + (log_prob_amt * is_bet)
        
        return cat_action.item(), amt_action.item(), total_log_prob
