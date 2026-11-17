"""
Proximal Policy Optimization (PPO) Self-Play Training Pipeline for Bagh-Chal.

Implements:
- Clipped surrogate objective with GAE advantage estimation
- Legal action masking throughout policy optimization
- Self-play rollout generation across asymmetric roles
- Training logging and model checkpointing
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.environment.board import TIGER, GOAT
from src.environment.state import TOTAL_ACTIONS, CANONICAL_ACTIONS, ACTION_TO_INDEX
from src.environment.env import BaghChalEnv
from src.training.networks import (
    MLPPolicyValueNet,
    CNNPolicyValueNet,
    GNNPolicyValueNet,
    DualNetworkWrapper,
    apply_action_mask,
)


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.gnn_nodes = []
        self.gnn_globals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.dones = []
        self.roles = []

    def clear(self):
        self.observations.clear()
        self.gnn_nodes.clear()
        self.gnn_globals.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.dones.clear()
        self.roles.clear()


class PPOTrainer:
    def __init__(
        self,
        representation: str = "flat",  # "flat" | "cnn" | "gnn"
        architecture: str = "single",   # "single" | "dual"
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
        save_dir: str = "data/checkpoints",
    ):
        self.representation = representation
        self.architecture = architecture
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        if device is None:
            self.device = os.getenv("BAGH_CHAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize neural network models
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def _build_single_network(self) -> nn.Module:
        if self.representation == "flat":
            return MLPPolicyValueNet(in_features=78, action_dim=TOTAL_ACTIONS)
        elif self.representation == "cnn":
            return CNNPolicyValueNet(in_channels=6, action_dim=TOTAL_ACTIONS)
        elif self.representation == "gnn":
            return GNNPolicyValueNet(node_features_dim=5, global_features_dim=4, action_dim=TOTAL_ACTIONS)
        else:
            raise ValueError(f"Unknown representation {self.representation}")

    def _build_model(self) -> nn.Module:
        if self.architecture == "single":
            return self._build_single_network().to(self.device)
        else:
            tiger_net = self._build_single_network()
            goat_net = self._build_single_network()
            return DualNetworkWrapper(tiger_net, goat_net).to(self.device)

    def _get_obs_and_forward(
        self, env: BaghChalEnv, mask_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        role = env.state.turn
        if self.representation == "flat":
            obs_np = env.get_flat_observation()
            obs_tensor = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)
            if self.architecture == "single":
                logits, value = self.model(obs_tensor, action_mask=mask_tensor)
            else:
                logits, value = self.model.get_network_for_role(role)(obs_tensor, action_mask=mask_tensor)
            return logits, value, obs_np

        elif self.representation == "cnn":
            obs_np = env.get_cnn_observation()
            obs_tensor = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)
            if self.architecture == "single":
                logits, value = self.model(obs_tensor, action_mask=mask_tensor)
            else:
                logits, value = self.model.get_network_for_role(role)(obs_tensor, action_mask=mask_tensor)
            return logits, value, obs_np

        else: # gnn
            gnn_data = env.get_gnn_observation()
            node_tensor = torch.from_numpy(gnn_data["node_features"]).to(self.device).unsqueeze(0)
            glob_tensor = torch.from_numpy(gnn_data["global_features"]).to(self.device).unsqueeze(0)
            if self.architecture == "single":
                logits, value = self.model(node_tensor, glob_tensor, action_mask=mask_tensor)
            else:
                logits, value = self.model.get_network_for_role(role)(node_tensor, glob_tensor, action_mask=mask_tensor)
            return logits, value, gnn_data

    def collect_episodes(self, env: BaghChalEnv, num_episodes: int = 10) -> Dict[str, float]:
        """
        Run self-play games to populate rollout buffer.
        """
        self.buffer.clear()
        total_tiger_wins = 0
        total_goat_wins = 0
        total_draws = 0

        for _ in range(num_episodes):
            env.reset()
            done = False

            while not done:
                legal = env.get_legal_actions()
                mask_np = env.get_action_mask()
                mask_tensor = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)
                role = env.state.turn

                with torch.no_grad():
                    logits, value, raw_obs = self._get_obs_and_forward(env, mask_tensor)
                    logits = logits.squeeze(0)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()

                act = CANONICAL_ACTIONS[action_idx]
                _, reward, term, trunc, info = env.step(act)
                done = term or trunc

                # Store transition
                if self.representation in ["flat", "cnn"]:
                    self.buffer.observations.append(raw_obs)
                else:
                    self.buffer.gnn_nodes.append(raw_obs["node_features"])
                    self.buffer.gnn_globals.append(raw_obs["global_features"])

                self.buffer.actions.append(action_idx)
                self.buffer.log_probs.append(log_prob)
                self.buffer.rewards.append(reward)
                self.buffer.values.append(value.item())
                self.buffer.masks.append(mask_np)
                self.buffer.dones.append(done)
                self.buffer.roles.append(role)

            if info["winner"] == TIGER:
                total_tiger_wins += 1
            elif info["winner"] == GOAT:
                total_goat_wins += 1
            else:
                total_draws += 1

        return {
            "tiger_win_rate": total_tiger_wins / num_episodes,
            "goat_win_rate": total_goat_wins / num_episodes,
            "draw_rate": total_draws / num_episodes,
            "total_steps": len(self.buffer.actions),
        }

    def train_step(self) -> Dict[str, float]:
        """
        Compute GAE and perform PPO policy & value network updates.
        """
        if len(self.buffer.rewards) == 0:
            return {}

        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        n_steps = len(rewards)

        # Compute Generalized Advantage Estimation (GAE)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            next_val = values[t + 1] if t + 1 < n_steps and not dones[t] else 0.0
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert to PyTorch tensors
        actions_t = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)
        roles_t = torch.tensor(self.buffer.roles, dtype=torch.long, device=self.device)

        if self.representation in ["flat", "cnn"]:
            obs_t = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        else:
            gnn_nodes_t = torch.tensor(np.array(self.buffer.gnn_nodes), dtype=torch.float32, device=self.device)
            gnn_globals_t = torch.tensor(np.array(self.buffer.gnn_globals), dtype=torch.float32, device=self.device)

        total_loss_accum = 0.0
        total_p_loss_accum = 0.0
        total_v_loss_accum = 0.0
        total_entropy_accum = 0.0

        indices = np.arange(n_steps)
        update_iters = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, self.batch_size):
                end = min(start + self.batch_size, n_steps)
                batch_idx = indices[start:end]

                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_adv = adv_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_masks = masks_t[batch_idx]
                b_roles = roles_t[batch_idx]

                if self.architecture == "dual":
                    if self.representation in ["flat", "cnn"]:
                        b_obs = obs_t[batch_idx]
                        logits, values = self.model(b_obs, role=b_roles, action_mask=b_masks)
                    else:
                        b_nodes = gnn_nodes_t[batch_idx]
                        b_globals = gnn_globals_t[batch_idx]
                        logits, values = self.model(
                            {"node_features": b_nodes, "global_features": b_globals},
                            role=b_roles,
                            action_mask=b_masks,
                        )
                else:
                    if self.representation in ["flat", "cnn"]:
                        b_obs = obs_t[batch_idx]
                        logits, values = self.model(b_obs, action_mask=b_masks)
                    else:
                        b_nodes = gnn_nodes_t[batch_idx]
                        b_globals = gnn_globals_t[batch_idx]
                        logits, values = self.model(b_nodes, b_globals, action_mask=b_masks)

                values = values.squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate loss
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values, b_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_accum += loss.item()
                total_p_loss_accum += policy_loss.item()
                total_v_loss_accum += value_loss.item()
                total_entropy_accum += entropy.item()
                update_iters += 1

        k = max(1, update_iters)
        return {
            "loss": total_loss_accum / k,
            "policy_loss": total_p_loss_accum / k,
            "value_loss": total_v_loss_accum / k,
            "entropy": total_entropy_accum / k,
        }

    def save_checkpoint(self, filename: str) -> str:
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        return path

