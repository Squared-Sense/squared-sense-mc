"""
Vectorized Parallel Environments for Squared Sense MC Offline Training.

Wraps MinecraftPathfindingEnv in Gymnasium VectorEnv (AsyncVectorEnv / SyncVectorEnv)
for 8–32 parallel workers reading from shared read-only .memmap files with minimal RAM overhead.
Optionally adapts observations to the 16-channel World Model tensor schema.
"""

import glob
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Callable, Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from offline.adapter import ObservationAdapter
from offline.lod_extractor import WorldVolume
from offline.palette import BlockPalette, get_palette
from offline.pathfinding_env import MinecraftPathfindingEnv
from offline.spawn_indexer import SpawnSampler


class AdaptedPathfindingWrapper(gym.Wrapper):
    """
    Gymnasium wrapper around MinecraftPathfindingEnv that automatically translates
    raw block observations into the strict 16-channel float32 (3, 16, 16, 16, 16)
    tensor representation matching docs/WORLD_MODEL_CONTRACTS.md.
    """

    def __init__(self, env: MinecraftPathfindingEnv, adapter: Optional[ObservationAdapter] = None):
        super().__init__(env)
        self.adapter = adapter or ObservationAdapter(palette=env.palette)

        # 16-Channel observation space
        self.observation_space = spaces.Dict({
            "shells": spaces.Box(low=0.0, high=1.0, shape=(3, 16, 16, 16, 16), dtype=np.float32),
            "state": spaces.Box(low=-100.0, high=100.0, shape=(12,), dtype=np.float32),
            "goal_vec": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "goal_dist": spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(env.num_actions,), dtype=np.bool_)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._adapt(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._adapt(obs), reward, term, trunc, info

    def _adapt(self, raw_obs: Dict) -> Dict[str, np.ndarray]:
        adapted = self.adapter.adapt(
            raw_obs=raw_obs,
            player_pos=self.env.player_pos,
            target_pos=self.env.target_pos,
            cardinal=self.env.cardinal,
            camera_pitch=self.env.camera_pitch,
            prev_action=self.env.prev_action
        )
        return {
            "shells": adapted["shells"],
            "state": adapted["state"],
            "goal_vec": adapted["goal_vec"],
            "goal_dist": np.array([adapted["goal_dist"]], dtype=np.float32),
            "action_mask": raw_obs.get("action_mask", self.env.get_action_mask())
        }


def make_pathfinding_env(
    region_dir: str,
    spawns_path: Optional[str] = None,
    allow_turns: bool = True,
    mask_fatal: bool = True,
    adapt_to_world_model: bool = True,
    seed: Optional[int] = None,
    world: Optional[WorldVolume] = None,
    palette: Optional[BlockPalette] = None,
    sampler: Optional[SpawnSampler] = None
) -> gym.Env:
    """
    Factory creating a single MinecraftPathfindingEnv instance with shared memmap access.
    """
    palette_obj = palette or get_palette()
    world_obj = world or WorldVolume(region_dir)

    if sampler is None:
        if spawns_path is None:
            # Find first spawns.npz in region directory
            candidates = glob.glob(os.path.join(region_dir, "*.spawns.npz"))
            if not candidates:
                raise FileNotFoundError(f"No .spawns.npz files found in {region_dir}")
            spawns_path = candidates[0]
        sampler_obj = SpawnSampler.from_npz(spawns_path)
    else:
        sampler_obj = sampler

    env = MinecraftPathfindingEnv(
        world_volume=world_obj,
        spawn_sampler=sampler_obj,
        palette_obj=palette_obj,
        allow_turns=allow_turns,
        mask_fatal=mask_fatal
    )

    if seed is not None:
        env.reset(seed=seed)

    if adapt_to_world_model:
        return AdaptedPathfindingWrapper(env)
    return env


def make_vector_env(
    num_envs: int = 8,
    region_dir: str = "baked_regions_test",
    spawns_path: Optional[str] = None,
    allow_turns: bool = True,
    mask_fatal: bool = True,
    adapt_to_world_model: bool = True,
    async_vector: bool = False,
    base_seed: int = 42
) -> gym.vector.VectorEnv:
    """
    Creates a vectorized parallel environment (SyncVectorEnv or AsyncVectorEnv).

    Args:
        num_envs: Number of parallel environment instances (e.g., 4, 8, 16, 32).
        region_dir: Directory containing baked .mmap regions.
        spawns_path: Optional path to .spawns.npz.
        allow_turns: If True, action space is 6 (cardinal turns); if False, 4 (pure translation).
        mask_fatal: If True, action mask disables moves leading to lethal falls/hazards.
        adapt_to_world_model: If True, wraps with 16-channel AdaptedPathfindingWrapper.
        async_vector: If True, uses AsyncVectorEnv (multiprocess); if False, SyncVectorEnv.
        base_seed: Base seed for deterministic PRNG across parallel workers.

    Returns:
        gym.vector.VectorEnv instance.
    """
    if spawns_path is None:
        candidates = glob.glob(os.path.join(region_dir, "*.spawns.npz"))
        if not candidates:
            raise FileNotFoundError(f"No .spawns.npz files found in {region_dir}")
        spawns_path = candidates[0]

    def create_thunk(rank: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            return make_pathfinding_env(
                region_dir=region_dir,
                spawns_path=spawns_path,
                allow_turns=allow_turns,
                mask_fatal=mask_fatal,
                adapt_to_world_model=adapt_to_world_model,
                seed=base_seed + rank
            )
        return _thunk

    env_fns = [create_thunk(i) for i in range(num_envs)]

    if async_vector:
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)
