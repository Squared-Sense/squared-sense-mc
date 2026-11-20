"""
Parallel PPO Training Engine & Behavioral Cloning Warm-Start for Squared Sense MC.

Trains an Actor-Critic 3D Perception & Navigation policy on parallel vectorized
environments reading directly from shared memory-mapped world volumes.
Features:
  - Siamese Conv3D multi-LOD encoder + CoordConv/Proprioception fusion
  - Masked action policy (invalid/fatal actions masked with -inf)
  - Generalized Advantage Estimation (GAE-Lambda)
  - Clipped PPO surrogate objective + Value function + Entropy regularization
  - Behavioral Cloning (BC) warm-start against optimal BFS paths
  - Periodic checkpointing (.pt + .json) and structured telemetry logging
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from offline.vector_env import make_vector_env


# ---------------------------------------------------------------------------
# Neural Network Architectures
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class ConvNeXt3DSiameseBlock(nn.Module):
        """Depthwise separable 3D convolutional block with residual connection."""

        def __init__(self, dim: int):
            super().__init__()
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
            self.norm = nn.LayerNorm(dim)
            self.pwconv1 = nn.Linear(dim, 4 * dim)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (N, C, D, H, W)
            residual = x
            x = self.dwconv(x)
            # Permute for LayerNorm over channel dimension: (N, D, H, W, C)
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = x.permute(0, 4, 1, 2, 3)
            return residual + x


    class VoxelPerceptionEncoder(nn.Module):
        """
        Siamese 3D ConvNeXt encoder for 3 concentric multi-LOD voxel shells.
        Encodes (B, 3, 16, 16, 16, 16) into 384-dimensional spatial feature vector.
        """

        def __init__(self, in_channels: int = 16, lod_feature_dim: int = 128):
            super().__init__()
            self.in_channels = in_channels
            self.lod_feature_dim = lod_feature_dim

            # Shared stem for each LOD: (B*3, 16, 16, 16, 16) -> (B*3, 32, 8, 8, 8)
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=2, stride=2),
                nn.GELU(),
                ConvNeXt3DSiameseBlock(32)
            )

            # Downsample: (B*3, 32, 8, 8, 8) -> (B*3, 64, 4, 4, 4)
            self.down1 = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=2, stride=2),
                nn.GELU(),
                ConvNeXt3DSiameseBlock(64)
            )

            # Head: (B*3, 64, 4, 4, 4) -> (B*3, lod_feature_dim, 1, 1, 1)
            self.head = nn.Sequential(
                nn.Conv3d(64, lod_feature_dim, kernel_size=4, stride=1),
                nn.GELU()
            )

        def forward(self, shells: torch.Tensor) -> torch.Tensor:
            """
            Args:
                shells: (B, 3, 16, 16, 16, 16) float32 channel-last (B, LOD, X, Y, Z, C).

            Returns:
                Combined features of shape (B, 3 * lod_feature_dim = 384).
            """
            B, num_lods, X, Y, Z, C = shells.shape
            # Reshape to (B * 3, C, X, Y, Z) for 3D convolution
            x = shells.permute(0, 1, 5, 2, 3, 4).reshape(B * num_lods, C, X, Y, Z)

            h = self.stem(x)
            h = self.down1(h)
            h = self.head(h)  # (B * 3, lod_feature_dim, 1, 1, 1)

            h = h.squeeze(-1).squeeze(-1).squeeze(-1)  # (B * 3, lod_feature_dim)
            h = h.reshape(B, num_lods * self.lod_feature_dim)  # (B, 384)
            return h


    class VoxelActorCritic(nn.Module):
        """
        Complete Actor-Critic Architecture with 3D perception and action masking.
        """

        def __init__(
            self,
            num_actions: int = 6,
            in_channels: int = 16,
            lod_feature_dim: int = 128,
            proprio_dim: int = 12,
            goal_dim: int = 3,
            hidden_dim: int = 128
        ):
            super().__init__()
            self.num_actions = num_actions
            self.encoder = VoxelPerceptionEncoder(in_channels=in_channels, lod_feature_dim=lod_feature_dim)

            # Proprioception + Goal vector MLP
            spatial_dim = 3 * lod_feature_dim  # 384
            aux_dim = proprio_dim + goal_dim   # 15
            self.aux_mlp = nn.Sequential(
                nn.Linear(aux_dim, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU()
            )

            # Policy (Actor) Head
            fused_dim = spatial_dim + 64  # 448
            self.actor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_actions)
            )

            # Value (Critic) Head
            self.critic = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward_features(
            self,
            shells: torch.Tensor,
            state: torch.Tensor,
            goal_vec: torch.Tensor
        ) -> torch.Tensor:
            spatial_feat = self.encoder(shells)
            aux_in = torch.cat([state, goal_vec], dim=-1)
            aux_feat = self.aux_mlp(aux_in)
            return torch.cat([spatial_feat, aux_feat], dim=-1)

        def get_value(
            self,
            shells: torch.Tensor,
            state: torch.Tensor,
            goal_vec: torch.Tensor
        ) -> torch.Tensor:
            feat = self.forward_features(shells, state, goal_vec)
            return self.critic(feat).squeeze(-1)

        def get_action_and_value(
            self,
            shells: torch.Tensor,
            state: torch.Tensor,
            goal_vec: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Computes action, log_prob, entropy, and value estimate.
            """
            feat = self.forward_features(shells, state, goal_vec)
            logits = self.actor(feat)

            if action_mask is not None:
                # Mask invalid moves; if all are masked, fallback to allow all to avoid NaN
                mask_bool = action_mask.bool()
                all_masked = ~mask_bool.any(dim=-1, keepdim=True)
                safe_mask = mask_bool | all_masked
                logits = torch.where(safe_mask, logits, torch.tensor(-1e9, device=logits.device))

            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.critic(feat).squeeze(-1)

            return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# GAE Advantage & Discounted Returns Calculator
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Generalized Advantage Estimation across rollout steps.

    Args:
        rewards: (T, N) float array.
        values: (T, N) float array.
        dones: (T, N) boolean array.
        next_value: (N,) float array for bootstrap state.

    Returns:
        advantages (T, N) and returns (T, N).
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Trainer Class
# ---------------------------------------------------------------------------

class ParallelPpoTrainer:
    """
    High-performance parallel PPO trainer operating on vectorized environments.
    """

    def __init__(
        self,
        num_envs: int = 8,
        region_dir: str = "baked_regions_test",
        lr: float = 3e-4,
        rollout_steps: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to run ParallelPpoTrainer.")

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1. Initialize Vectorized Environment
        self.envs = make_vector_env(
            num_envs=num_envs,
            region_dir=region_dir,
            allow_turns=True,
            mask_fatal=True,
            adapt_to_world_model=True,
            async_vector=False
        )

        # 2. Initialize Model & Optimizer
        num_actions = self.envs.single_action_space.n
        self.model = VoxelActorCritic(num_actions=num_actions).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, eps=1e-5)

        # 3. Training Telemetry History
        self.history = {
            "step": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "success_rate": [],
            "efficiency_ratio": [],
            "mean_return": []
        }

    def train_step(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Collects one rollout across all parallel workers and updates policy."""
        T = self.rollout_steps
        N = self.num_envs

        # Rollout Storage Buffers
        obs_shells = np.zeros((T, N, 3, 16, 16, 16, 16), dtype=np.float32)
        obs_states = np.zeros((T, N, 12), dtype=np.float32)
        obs_goals = np.zeros((T, N, 3), dtype=np.float32)
        obs_masks = np.zeros((T, N, self.model.num_actions), dtype=bool)
        actions = np.zeros((T, N), dtype=np.int64)
        log_probs = np.zeros((T, N), dtype=np.float32)
        rewards = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=bool)
        values = np.zeros((T, N), dtype=np.float32)

        # Reset / Initial Step
        if not hasattr(self, "current_obs"):
            self.current_obs, _ = self.envs.reset()

        ep_successes = []
        ep_efficiencies = []
        ep_returns = []

        # 1. Collect Rollout
        for t in range(T):
            obs_shells[t] = self.current_obs["shells"]
            obs_states[t] = self.current_obs["state"]
            obs_goals[t] = self.current_obs["goal_vec"]
            obs_masks[t] = self.current_obs["action_mask"]

            with torch.no_grad():
                s_t = torch.from_numpy(obs_shells[t]).to(self.device)
                st_t = torch.from_numpy(obs_states[t]).to(self.device)
                g_t = torch.from_numpy(obs_goals[t]).to(self.device)
                m_t = torch.from_numpy(obs_masks[t]).to(self.device)

                act, lp, _, val = self.model.get_action_and_value(s_t, st_t, g_t, m_t)

            actions[t] = act.cpu().numpy()
            log_probs[t] = lp.cpu().numpy()
            values[t] = val.cpu().numpy()

            next_obs, step_rew, term, trunc, infos = self.envs.step(actions[t])
            rewards[t] = step_rew
            dones[t] = term | trunc
            self.current_obs = next_obs

            # Parse episode info
            if "final_info" in infos:
                for fin in infos["final_info"]:
                    if fin is not None:
                        if "is_success" in fin:
                            ep_successes.append(float(fin["is_success"]))
                        if "efficiency_ratio" in fin:
                            ep_efficiencies.append(float(fin["efficiency_ratio"]))
                        if "episode_return" in fin:
                            ep_returns.append(float(fin["episode_return"]))

        # 2. Bootstrap Next Value
        with torch.no_grad():
            next_s = torch.from_numpy(self.current_obs["shells"]).to(self.device)
            next_st = torch.from_numpy(self.current_obs["state"]).to(self.device)
            next_g = torch.from_numpy(self.current_obs["goal_vec"]).to(self.device)
            next_val = self.model.get_value(next_s, next_st, next_g).cpu().numpy()

        # 3. Compute GAE Advantages
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_val,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # 4. Flatten Rollouts for Optimization
        b_shells = torch.from_numpy(obs_shells.reshape(-1, 3, 16, 16, 16, 16)).to(self.device)
        b_states = torch.from_numpy(obs_states.reshape(-1, 12)).to(self.device)
        b_goals = torch.from_numpy(obs_goals.reshape(-1, 3)).to(self.device)
        b_masks = torch.from_numpy(obs_masks.reshape(-1, self.model.num_actions)).to(self.device)
        b_actions = torch.from_numpy(actions.reshape(-1)).to(self.device)
        b_log_probs = torch.from_numpy(log_probs.reshape(-1)).to(self.device)
        b_advantages = torch.from_numpy(advantages.reshape(-1)).to(self.device)
        b_returns = torch.from_numpy(returns.reshape(-1)).to(self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = T * N
        indices = np.arange(total_samples)

        epoch_p_loss = []
        epoch_v_loss = []
        epoch_ent = []

        # 5. PPO Update Epochs
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                _, new_lp, entropy, new_val = self.model.get_action_and_value(
                    b_shells[mb_idx],
                    b_states[mb_idx],
                    b_goals[mb_idx],
                    b_masks[mb_idx],
                    b_actions[mb_idx]
                )

                log_ratio = new_lp - b_log_probs[mb_idx]
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                mb_adv = b_advantages[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_loss = 0.5 * ((new_val - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_p_loss.append(pg_loss.item())
                epoch_v_loss.append(v_loss.item())
                epoch_ent.append(ent_loss.item())

        metrics = {
            "policy_loss": float(np.mean(epoch_p_loss)),
            "value_loss": float(np.mean(epoch_v_loss)),
            "entropy": float(np.mean(epoch_ent)),
            "success_rate": float(np.mean(ep_successes)) if ep_successes else 0.0,
            "efficiency_ratio": float(np.mean(ep_efficiencies)) if ep_efficiencies else 1.0,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else float(rewards.sum() / N)
        }

        # Log history
        step_idx = len(self.history["step"]) + 1
        self.history["step"].append(step_idx * total_samples)
        self.history["policy_loss"].append(metrics["policy_loss"])
        self.history["value_loss"].append(metrics["value_loss"])
        self.history["entropy"].append(metrics["entropy"])
        self.history["success_rate"].append(metrics["success_rate"])
        self.history["efficiency_ratio"].append(metrics["efficiency_ratio"])
        self.history["mean_return"].append(metrics["mean_return"])

        return metrics

    def save_checkpoint(self, filename: str = "ppo_checkpoint.pt") -> str:
        """Saves model weights and training telemetry metadata."""
        path = os.path.join(self.checkpoint_dir, filename)
        meta_path = path.replace(".pt", ".json")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_actions": self.model.num_actions,
            "history": self.history
        }, path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "saved_at": time.time(),
                "steps": self.history["step"][-1] if self.history["step"] else 0,
                "latest_policy_loss": self.history["policy_loss"][-1] if self.history["policy_loss"] else 0.0,
                "latest_value_loss": self.history["value_loss"][-1] if self.history["value_loss"] else 0.0,
                "latest_success_rate": self.history["success_rate"][-1] if self.history["success_rate"] else 0.0,
                "history": self.history
            }, f, indent=4)

        return path

    def load_checkpoint(self, path: str):
        """Loads model weights and history from checkpoint file."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        if "optimizer_state_dict" in data:
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
        if "history" in data:
            self.history = data["history"]
        print(f"[ParallelPpoTrainer] Loaded checkpoint from {path}")
