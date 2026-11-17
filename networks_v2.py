# -*- coding: utf-8 -*-
"""FC(256) 基线与可选时间/因子 Attention 的共享 Actor-Critic。"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        normalized = self.norm1(x)
        attended, weights = self.attention(
            normalized, normalized, normalized, need_weights=need_weights,
            average_attn_weights=True,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, weights


class AttentionActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lookback = int(config.get("lookback", 60))
        self.feature_count = int(config.get("feature_count", 20))
        self.account_count = int(config.get("account_count", 8))
        self.use_attention = bool(config.get("use_attention", False))
        self.state_dim = self.lookback * self.feature_count + self.account_count
        d_model = int(config.get("d_model", 64))
        heads = int(config.get("heads", 4))
        layers = int(config.get("layers", 2))
        hidden = int(config.get("fc_hidden", 256))
        dropout = float(config.get("dropout", 0.1))

        shared_input = self.state_dim
        if self.use_attention:
            shared_input = d_model * 3
            self.input_projection = nn.Linear(self.feature_count, d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, self.lookback, d_model))
            self.time_layers = nn.ModuleList([
                AttentionEncoderLayer(d_model, heads, dropout) for _ in range(layers)
            ])
            self.factor_value_embedding = nn.Linear(1, d_model)
            self.factor_identity = nn.Parameter(torch.zeros(1, self.feature_count, d_model))
            self.factor_attention = nn.MultiheadAttention(
                d_model, heads, dropout=dropout, batch_first=True
            )
            self.factor_norm = nn.LayerNorm(d_model)
            self.account_projection = nn.Linear(self.account_count, d_model)
            nn.init.normal_(self.position_embedding, std=0.02)
            nn.init.normal_(self.factor_identity, std=0.02)

        self.shared = nn.Sequential(
            nn.Linear(shared_input, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((1,), float(config.get("log_std_init", -0.7))))

    def forward(self, flat_state, return_attention=False):
        if flat_state.ndim == 1:
            flat_state = flat_state.unsqueeze(0)
        time_weights = None
        factor_weights = None
        if self.use_attention:
            batch = flat_state.shape[0]
            sequence_size = self.lookback * self.feature_count
            sequence = flat_state[:, :sequence_size].reshape(batch, self.lookback, self.feature_count)
            account = flat_state[:, sequence_size:sequence_size + self.account_count]
            time_tokens = self.input_projection(sequence) + self.position_embedding
            for layer in self.time_layers:
                time_tokens, time_weights = layer(time_tokens, need_weights=return_attention)
            time_summary = time_tokens.mean(dim=1)
            latest_factors = sequence[:, -1, :].unsqueeze(-1)
            factor_tokens = self.factor_value_embedding(latest_factors) + self.factor_identity
            normalized_factors = self.factor_norm(factor_tokens)
            factor_attended, factor_weights = self.factor_attention(
                normalized_factors, normalized_factors, normalized_factors,
                need_weights=return_attention, average_attn_weights=True,
            )
            factor_summary = (factor_tokens + factor_attended).mean(dim=1)
            account_summary = self.account_projection(account)
            shared = self.shared(torch.cat([time_summary, factor_summary, account_summary], dim=-1))
        else:
            shared = self.shared(flat_state)

        mean = self.actor(shared).squeeze(-1)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        value = self.critic(shared).squeeze(-1)
        if return_attention:
            return mean, std, value, {"time": time_weights, "feature": factor_weights}
        return mean, std, value


class PPOAgent:
    def __init__(self, network_config, ppo_config, device):
        self.device = torch.device(device)
        self.policy = AttentionActorCritic(network_config).to(self.device)
        learning_rate = float(ppo_config.get("learning_rate", 1e-4))
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, weight_decay=1e-5)

    def select_action(self, state, deterministic=False):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.policy(tensor)
            distribution = Normal(mean, std)
            action = mean if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
        return (
            float(action.cpu().numpy().reshape(-1)[0]),
            float(log_prob.cpu().item()),
            float(value.cpu().item()),
        )

    def value(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.policy(tensor)[2].cpu().item())

    @torch.no_grad()
    def attention_weights(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, _, weights = self.policy(tensor, return_attention=True)
        return {
            key: value[0].detach().cpu().numpy() if value is not None else None
            for key, value in weights.items()
        }
