"""
Unit & Integration Tests for Phase 4: Parallel Training Pipeline & Interoperability Adapter.
"""

import os
import unittest
import numpy as np
import torch

from offline.adapter import (
    CH_AIR,
    CH_TERRAIN,
    CH_WAYPOINT,
    ObservationAdapter
)
from offline.lod_extractor import WorldVolume
from offline.palette import get_palette
from offline.pathfinding_env import MinecraftPathfindingEnv
from offline.spawn_indexer import SpawnSampler
from offline.train_ppo import ParallelPpoTrainer, VoxelActorCritic, compute_gae
from offline.vector_env import make_vector_env


class TestPhase4ParallelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "baked_regions_test"
        if not os.path.exists(cls.test_dir):
            raise unittest.SkipTest("baked_regions_test not found")

        cls.world = WorldVolume(cls.test_dir)
        cls.palette = get_palette()
        cls.sampler = SpawnSampler.from_npz(os.path.join(cls.test_dir, "r.-1.2.spawns.npz"))
        cls.env = MinecraftPathfindingEnv(cls.world, cls.sampler, palette_obj=cls.palette)
        cls.adapter = ObservationAdapter(cls.palette)

    def test_observation_adapter_16_channels(self):
        """Verify ObservationAdapter produces exact 16-channel float32 tensors matching CONTRACT.md."""
        obs, info = self.env.reset(seed=123)
        adapted = self.adapter.adapt(
            raw_obs=obs,
            player_pos=self.env.player_pos,
            target_pos=self.env.target_pos,
            cardinal=self.env.cardinal,
            camera_pitch=self.env.camera_pitch,
            prev_action=self.env.prev_action
        )

        shells = adapted["shells"]
        state = adapted["state"]
        goal_vec = adapted["goal_vec"]

        # Tensor dimensions
        self.assertEqual(shells.shape, (3, 16, 16, 16, 16))
        self.assertEqual(shells.dtype, np.float32)
        self.assertEqual(state.shape, (12,))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(goal_vec.shape, (3,))
        self.assertEqual(goal_vec.dtype, np.float32)

        # Range bounds
        self.assertTrue(np.all(shells >= 0.0) and np.all(shells <= 1.0))
        goal_norm = np.linalg.norm(goal_vec)
        self.assertAlmostEqual(float(goal_norm), 1.0, places=5)

        # Verify Channel 8 (WAYPOINT) has active beacon
        # If target is within LOD bounds, waypoint channel should contain 1.0
        lod2_waypoints = shells[2, :, :, :, CH_WAYPOINT]
        self.assertGreater(lod2_waypoints.sum(), 0.0)

    def test_vector_environment_execution(self):
        """Verify parallel VectorEnv creates workers and executes steps synchronously."""
        num_envs = 4
        envs = make_vector_env(
            num_envs=num_envs,
            region_dir=self.test_dir,
            allow_turns=True,
            mask_fatal=True,
            adapt_to_world_model=True,
            async_vector=False,
            base_seed=100
        )

        obs, info = envs.reset()
        self.assertEqual(obs["shells"].shape, (num_envs, 3, 16, 16, 16, 16))
        self.assertEqual(obs["state"].shape, (num_envs, 12))
        self.assertEqual(obs["goal_vec"].shape, (num_envs, 3))
        self.assertEqual(obs["action_mask"].shape, (num_envs, 6))

        # Perform 3 batch steps
        for _ in range(3):
            # Sample random valid actions per environment
            actions = np.zeros(num_envs, dtype=np.int64)
            for i in range(num_envs):
                valid = np.where(obs["action_mask"][i])[0]
                actions[i] = np.random.choice(valid)

            next_obs, rewards, terms, truncs, infos = envs.step(actions)
            self.assertEqual(len(rewards), num_envs)
            self.assertEqual(len(terms), num_envs)
            obs = next_obs

        envs.close()

    def test_gae_advantage_calculation(self):
        """Verify Generalized Advantage Estimation calculation."""
        T, N = 10, 2
        rewards = np.full((T, N), 0.5, dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=bool)
        next_val = np.zeros(N, dtype=np.float32)

        advantages, returns = compute_gae(rewards, values, dones, next_val, gamma=0.99, gae_lambda=0.95)
        self.assertEqual(advantages.shape, (T, N))
        self.assertEqual(returns.shape, (T, N))
        self.assertTrue(np.all(advantages > 0.0))

    def test_ppo_actor_critic_and_checkpointing(self):
        """Verify VoxelActorCritic forward pass, PPO training step, and serialization."""
        checkpoint_dir = "test_checkpoints"
        trainer = ParallelPpoTrainer(
            num_envs=2,
            region_dir=self.test_dir,
            rollout_steps=4,
            checkpoint_dir=checkpoint_dir
        )

        metrics = trainer.train_step(num_epochs=1, batch_size=4)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)

        ckpt_path = trainer.save_checkpoint("phase4_test.pt")
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertTrue(os.path.exists(ckpt_path.replace(".pt", ".json")))

        # Test loading checkpoint back into model
        trainer.load_checkpoint(ckpt_path)


if __name__ == "__main__":
    unittest.main()

