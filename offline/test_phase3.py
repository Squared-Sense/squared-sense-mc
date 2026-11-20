"""
Unit & Integration Test Suite for Phase 3 (Gymnasium Pathfinding RL Environment).
"""

import math
import os
import unittest
import numpy as np

from offline.bfs_target import BoundedBfsTargetGenerator, compute_b2_bounding_box
from offline.curriculum import DistanceCurriculumScheduler
from offline.lod_extractor import GRID_SIZE, LodExtractor, WorldVolume
from offline.palette import get_palette
from offline.pathfinding_env import (
    ACTION_FORWARD,
    ACTION_NAMES,
    CARDINAL_VECTORS,
    MinecraftPathfindingEnv,
)
from offline.physics_kernel import MovementPhysicsKernel, MoveOutcome
from offline.spawn_indexer import SpawnSampler


class TestPhase3PhysicsAndBFS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "baked_regions_test"
        if not os.path.exists(cls.test_dir):
            raise unittest.SkipTest("baked_regions_test not found")
        cls.world = WorldVolume(cls.test_dir)
        cls.palette = get_palette()
        cls.kernel = MovementPhysicsKernel(cls.world, cls.palette)
        cls.sampler = SpawnSampler.from_npz(os.path.join(cls.test_dir, "r.-1.2.spawns.npz"))
        cls.target_gen = BoundedBfsTargetGenerator(cls.kernel)

    def test_physics_standing_clearance(self):
        # A valid sampled spawn must always be a valid standing position
        for _ in range(5):
            spawn = self.sampler.sample()
            self.assertTrue(self.kernel.is_valid_standing_position(spawn[0], spawn[1], spawn[2]))

    def test_b2_bounding_box_clamping(self):
        spawn = self.sampler.sample()
        for cardinal in ["SOUTH", "WEST", "NORTH", "EAST"]:
            min_wx, max_wx, min_wy, max_wy, min_wz, max_wz = compute_b2_bounding_box(spawn, cardinal)
            # The span in world must be consistent with 64 blocks
            self.assertTrue(max_wx >= min_wx)
            self.assertTrue(max_wy >= min_wy)
            self.assertTrue(max_wz >= min_wz)
            self.assertGreaterEqual(max_wy - min_wy, 60)

    def test_bfs_target_generation_and_reachability(self):
        spawn = self.sampler.sample()
        target_pos, l_bfs, path = self.target_gen.generate_target(
            spawn=spawn,
            cardinal="SOUTH",
            target_dist_range=(4, 10)
        )

        self.assertGreaterEqual(l_bfs, 0)
        self.assertEqual(path[0], spawn)
        self.assertEqual(path[-1], target_pos)
        self.assertEqual(len(path) - 1, l_bfs)

        # Step-by-step verification: following the optimal path with kernel.step MUST reach target
        curr = spawn
        for next_step in path[1:]:
            dx = next_step[0] - curr[0]
            dz = next_step[2] - curr[2]
            nx, ny, nz, outcome, _ = self.kernel.step(curr[0], curr[1], curr[2], dx, dz)
            self.assertIn(outcome, (MoveOutcome.SUCCESS, MoveOutcome.STEP_UP, MoveOutcome.FALL))
            self.assertEqual((nx, ny, nz), next_step)
            curr = (nx, ny, nz)

        self.assertEqual(curr, target_pos)

    def test_bfs_candidate_diversity(self):
        # Sample 10 targets from curriculum stage 1 (range 6..12)
        # Should have diverse distances rather than all locked to 12
        distances = []
        for i in range(10):
            spawn = self.sampler.sample()
            _, l_bfs, _ = self.target_gen.generate_target(
                spawn=spawn,
                cardinal="SOUTH",
                target_dist_range=(6, 12)
            )
            distances.append(l_bfs)

        unique_dists = set(distances)
        self.assertGreater(len(unique_dists), 1, "Target generator should produce diverse distances in range [6, 12]")


class TestPhase3CurriculumScheduler(unittest.TestCase):
    def test_stage_advancement(self):
        scheduler = DistanceCurriculumScheduler(
            initial_stage=0,
            window_size=10,
            advance_threshold=0.8,
            min_episodes_before_advance=10
        )
        self.assertEqual(scheduler.stage, 0)
        self.assertEqual(scheduler.get_distance_range(), (2, 6))

        # 10 successful episodes -> advance to stage 1
        for _ in range(10):
            scheduler.record_episode(success=True)

        self.assertEqual(scheduler.stage, 1)
        self.assertEqual(scheduler.get_distance_range(), (6, 12))


class TestPhase3GymnasiumEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "baked_regions_test"
        if not os.path.exists(cls.test_dir):
            raise unittest.SkipTest("baked_regions_test not found")
        cls.world = WorldVolume(cls.test_dir)
        cls.palette = get_palette()
        cls.sampler = SpawnSampler.from_npz(os.path.join(cls.test_dir, "r.-1.2.spawns.npz"))

    def test_env_reset_and_step(self):
        env = MinecraftPathfindingEnv(
            world_volume=self.world,
            spawn_sampler=self.sampler,
            palette_obj=self.palette,
            allow_turns=True
        )

        obs, info = env.reset(seed=42)

        # Check observation space compliance
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["goal_delta_xyz"].shape, (3,))
        self.assertEqual(obs["goal_dist"].shape, (1,))
        self.assertEqual(obs["prev_action"].shape, (1,))
        self.assertEqual(obs["action_mask"].shape, (6,))

        # Step through 5 random actions
        for _ in range(5):
            mask = obs["action_mask"]
            valid_actions = np.where(mask)[0]
            action = int(np.random.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            if terminated or truncated:
                break

        env.close()

    def test_oracle_agent_reaches_goal(self):
        """
        An agent following the BFS optimal reference path must reach the goal,
        trigger termination, and earn positive return.
        """
        env = MinecraftPathfindingEnv(
            world_volume=self.world,
            spawn_sampler=self.sampler,
            palette_obj=self.palette,
            allow_turns=False  # Pure 4-directional horizontal moves
        )

        obs, info = env.reset(seed=123)
        path = info["optimal_path"]
        l_bfs = info["l_bfs"]

        if l_bfs == 0:
            env.close()
            return

        # Follow path
        cardinal = env.cardinal
        vecs = CARDINAL_VECTORS[cardinal]
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        terminated = False
        step = 0
        while not terminated and step < len(path) - 1:
            curr = env.player_pos
            nxt = path[step + 1]
            dx = nxt[0] - curr[0]
            dz = nxt[2] - curr[2]

            # Map world (dx, dz) to action
            if (dx, dz) == (fx, fz):
                act = 0  # FORWARD
            elif (dx, dz) == (-fx, -fz):
                act = 1  # BACKWARD
            elif (dx, dz) == (-rx, -rz):
                act = 2  # LEFT
            elif (dx, dz) == (rx, rz):
                act = 3  # RIGHT
            else:
                break

            obs, reward, terminated, truncated, info = env.step(act)
            step += 1

        self.assertTrue(terminated, f"Oracle failed to terminate. Dist: {info['goal_dist']}, Steps: {step}")
        self.assertTrue(info["is_success"], "Oracle must reach the goal successfully")
        self.assertGreater(info["episode_return"], 0.0, "Oracle return must be positive")
        env.close()

    def test_action_mask_fatal(self):
        env_standard = MinecraftPathfindingEnv(
            world_volume=self.world,
            spawn_sampler=self.sampler,
            mask_fatal=False
        )
        env_fatal_masked = MinecraftPathfindingEnv(
            world_volume=self.world,
            spawn_sampler=self.sampler,
            mask_fatal=True
        )

        obs1, _ = env_standard.reset(seed=42)
        obs2, _ = env_fatal_masked.reset(seed=42)

        self.assertEqual(obs1["action_mask"].shape, (6,))
        self.assertEqual(obs2["action_mask"].shape, (6,))

        env_standard.close()
        env_fatal_masked.close()


if __name__ == "__main__":
    unittest.main()

