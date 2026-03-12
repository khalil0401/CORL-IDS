"""
sac_discrete.py

Module 4: Discrete Soft Actor-Critic

Components
----------
Actor    : maps state → categorical probability distribution over actions
Critic   : twin Q-networks (Q1, Q2) → Q(s, a) for all actions
Targets  : soft-updated target Q-networks
Alpha    : learnable entropy temperature (or fixed)

Actor Architecture:
    Linear(32 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → num_actions) → Softmax

Critic Architecture (twin):
    Linear(32 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → num_actions)

Update:
    - Critic loss : MSE( Q(s,a), r + γ * (min Q_target(s',a') - α * log π(a'|s')) )
    - Actor loss  : E[ α * log π(a|s) - min Q(s,a) ]
    - Alpha loss  : -α * (H(π) - H_target)    (auto-tuning; optional)
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
# Network building blocks
# ──────────────────────────────────────────────────────────────────────

def _make_mlp(input_dim: int, hidden1: int, hidden2: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1,  hidden2),
        nn.ReLU(),
        nn.Linear(hidden2,  output_dim),
    )


class DiscreteActor(nn.Module):
    """
    Categorical policy: outputs Softmax probability over all actions.
    """

    def __init__(self, state_dim: int, num_actions: int,
                 hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = _make_mlp(state_dim, hidden1, hidden2, num_actions)
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns probability distribution π(a|s)  shape (batch, num_actions)."""
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action, compute log-prob and entropy.

        Returns
        -------
        action     : (batch,)   int64
        log_pi     : (batch,)   log prob of sampled action
        entropy    : (batch,)   policy entropy  H = -Σ π log π
        probs      : (batch, num_actions)
        """
        probs = self.forward(state)
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        # Entropy: − Σ_a π(a|s) log π(a|s)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        return action, log_pi, entropy, probs

    def expand(self, new_num_actions: int):
        """Grow the output head to support additional action classes."""
        old_num = self.num_actions
        if new_num_actions <= old_num:
            return
        old_linear = list(self.net.children())[-1]
        new_linear  = nn.Linear(old_linear.in_features, new_num_actions)
        # Copy existing weights; init new rows with small random values
        with torch.no_grad():
            new_linear.weight[:old_num] = old_linear.weight
            new_linear.bias[:old_num]   = old_linear.bias
            nn.init.normal_(new_linear.weight[old_num:], std=0.01)
            nn.init.zeros_(new_linear.bias[old_num:])
        # Replace last layer
        net_list = list(self.net.children())[:-1] + [new_linear]
        self.net = nn.Sequential(*net_list)
        self.num_actions = new_num_actions


class DiscreteCritic(nn.Module):
    """
    Q-network: outputs Q(s, a) for all actions simultaneously.
    """

    def __init__(self, state_dim: int, num_actions: int,
                 hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = _make_mlp(state_dim, hidden1, hidden2, num_actions)
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values  shape (batch, num_actions)."""
        return self.net(state)

    def expand(self, new_num_actions: int):
        """Grow output head to support additional action classes."""
        old_num = self.num_actions
        if new_num_actions <= old_num:
            return
        old_linear = list(self.net.children())[-1]
        new_linear  = nn.Linear(old_linear.in_features, new_num_actions)
        with torch.no_grad():
            new_linear.weight[:old_num] = old_linear.weight
            new_linear.bias[:old_num]   = old_linear.bias
            nn.init.normal_(new_linear.weight[old_num:], std=0.01)
            nn.init.zeros_(new_linear.bias[old_num:])
        net_list  = list(self.net.children())[:-1] + [new_linear]
        self.net  = nn.Sequential(*net_list)
        self.num_actions = new_num_actions


# ──────────────────────────────────────────────────────────────────────
# SAC Agent
# ──────────────────────────────────────────────────────────────────────

class DiscreteSAC:
    """
    Discrete Soft Actor-Critic agent.

    Parameters
    ----------
    state_dim       : int    latent state dimension (32)
    num_actions     : int    initial number of discrete actions
    lr              : float  learning rate (default 3e-4)
    gamma           : float  discount factor (default 0.99)
    alpha           : float  fixed entropy temperature (set None for auto-tuning)
    tau             : float  soft update coefficient for target nets (default 0.005)
    device          : str
    """

    def __init__(self,
                 state_dim:   int,
                 num_actions: int,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 alpha:       float = 0.2,
                 tau:         float = 0.005,
                 auto_alpha:  bool  = True,
                 device:      str   = "cpu"):

        self.device      = torch.device(device)
        self.gamma       = gamma
        self.tau         = tau
        self.num_actions = num_actions

        # ── Actor ───────────────────────────────────────────────────
        self.actor = DiscreteActor(state_dim, num_actions).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        # ── Twin Critics ────────────────────────────────────────────
        self.q1 = DiscreteCritic(state_dim, num_actions).to(self.device)
        self.q2 = DiscreteCritic(state_dim, num_actions).to(self.device)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        # ── Target Critics (no grad) ─────────────────────────────────
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters(): p.requires_grad = False
        for p in self.q2_target.parameters(): p.requires_grad = False

        # ── Entropy temperature α ────────────────────────────────────
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -math.log(1.0 / num_actions) * 0.98
            self.log_alpha = torch.tensor(math.log(alpha),
                                          dtype=torch.float32,
                                          requires_grad=True,
                                          device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha     = alpha
            self.log_alpha = None
            self.alpha_optim = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action given a numpy state vector.

        Returns
        -------
        action : int
        probs  : np.ndarray (num_actions,)
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(s)                              # (1, num_actions)
        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            action = torch.distributions.Categorical(probs).sample().item()
        return action, probs.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # SAC update
    # ------------------------------------------------------------------

    def update(self,
               states:      torch.Tensor,
               actions:     torch.Tensor,
               rewards:     torch.Tensor,
               next_states: torch.Tensor,
               dones:       torch.Tensor) -> dict:
        """
        One gradient step on critics, actor, and alpha (if auto-tuning).

        All inputs are already on self.device as Tensors.

        Returns dict of scalar losses for logging.
        """
        alpha = self.alpha

        # ── Critic targets (no grad) ──────────────────────────────────
        with torch.no_grad():
            next_probs = self.actor(next_states)           # (B, A)
            next_log_pi = (next_probs + 1e-8).log()       # (B, A)

            q1_next = self.q1_target(next_states)          # (B, A)
            q2_next = self.q2_target(next_states)          # (B, A)
            min_q_next = torch.min(q1_next, q2_next)      # (B, A)

            # V(s') = Σ_a π(a|s') [ Q(s',a) - α log π(a|s') ]
            v_next = (next_probs * (min_q_next - alpha * next_log_pi)).sum(dim=-1)

            # Bellman target
            target_q = rewards + self.gamma * (1.0 - dones) * v_next  # (B,)

        # ── Critic update ─────────────────────────────────────────────
        q1_all = self.q1(states)                           # (B, A)
        q2_all = self.q2(states)                           # (B, A)

        # Gather Q-values for taken actions
        q1_taken = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        q2_taken = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = F.mse_loss(q1_taken, target_q) + F.mse_loss(q2_taken, target_q)

        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=5.0)
        self.q1_optim.step()
        self.q2_optim.step()

        # ── Actor update ──────────────────────────────────────────────
        probs    = self.actor(states)                      # (B, A)
        log_pi   = (probs + 1e-8).log()                   # (B, A)

        with torch.no_grad():
            min_q = torch.min(self.q1(states), self.q2(states))  # (B, A)

        # Actor loss = E_a[ α log π(a|s) - Q(s,a) ]
        actor_loss = (probs * (alpha * log_pi - min_q)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_optim.step()

        # ── Alpha update (auto-tune) ──────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            entropy = -(probs.detach() * log_pi.detach()).sum(dim=-1).mean()
            alpha_loss_val = -(self.log_alpha * (entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss_val.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_val.item()

        # ── Soft target update ────────────────────────────────────────
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ θ_source + (1-τ) θ_target"""
        for ps, pt in zip(source.parameters(), target.parameters()):
            pt.data.copy_(self.tau * ps.data + (1.0 - self.tau) * pt.data)

    # ------------------------------------------------------------------
    # Continual learning: expand action space
    # ------------------------------------------------------------------

    def expand_action_space(self, new_num_actions: int, lr: float = 3e-4):
        """
        Grow all networks to support new_num_actions.
        Re-creates optimisers after expansion (new parameters added).
        """
        if new_num_actions <= self.num_actions:
            return

        self.actor.expand(new_num_actions)
        self.q1.expand(new_num_actions)
        self.q2.expand(new_num_actions)
        self.q1_target.expand(new_num_actions)
        self.q2_target.expand(new_num_actions)

        # Sync targets to current nets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Rebuild optimisers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optim    = optim.Adam(self.q1.parameters(),    lr=lr)
        self.q2_optim    = optim.Adam(self.q2.parameters(),    lr=lr)

        if self.auto_alpha:
            self.target_entropy = -math.log(1.0 / new_num_actions) * 0.98

        self.num_actions = new_num_actions
        print(f"[SAC] Action space expanded to {new_num_actions}")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def state_dict_full(self) -> dict:
        return {
            "actor":       self.actor.state_dict(),
            "q1":          self.q1.state_dict(),
            "q2":          self.q2.state_dict(),
            "q1_target":   self.q1_target.state_dict(),
            "q2_target":   self.q2_target.state_dict(),
            "alpha":       self.alpha,
            "log_alpha":   self.log_alpha.item() if self.log_alpha is not None else None,
            "num_actions": self.num_actions,
        }

    def load_state_dict_full(self, ckpt: dict):
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.alpha = ckpt["alpha"]
        if self.log_alpha is not None and ckpt["log_alpha"] is not None:
            self.log_alpha.data.fill_(ckpt["log_alpha"])
