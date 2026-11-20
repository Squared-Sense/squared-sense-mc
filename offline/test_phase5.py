"""
Unit & Integration Tests for Phase 5: Single-Inference Mode & Visualizer Telemetry.
"""

import os
import unittest
import numpy as np

from bridge.mc_ipc_reader import MinecraftIpcReader
from offline.adapter import ObservationAdapter
from offline.ipc_emitter import MinecraftIpcEmitter
from offline.ipc_telemetry import InferenceTelemetry
from offline.lod_extractor import WorldVolume
from offline.palette import get_palette
from offline.pathfinding_env import MinecraftPathfindingEnv
from offline.run_inference import run_inference_episode
from offline.spawn_indexer import SpawnSampler


class TestPhase5InferenceAndTelemetry(unittest.TestCase):
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

    def test_telemetry_pack_and_unpack(self):
        """Verify binary packing and unpacking of inference telemetry."""
        target_pos = (120, 64, -200)
        player_pos = (115, 64, -205)
        target_rel = (5.0, 0.0, 5.0)
        remaining_dist = 7.07
        step_reward = 0.25
        episode_return = 3.50
        step_count = 14
        critic_value = 8.45
        action_probs = [0.70, 0.05, 0.05, 0.10, 0.05, 0.05]
        chosen_action = 0
        l_bfs = 12
        visited_trail = [(115, 64, -205), (116, 64, -205), (117, 64, -204)]
        bfs_path = [(115, 64, -205), (116, 64, -204), (118, 64, -202), (120, 64, -200)]

        packed_bytes = InferenceTelemetry.pack(
            target_pos=target_pos,
            player_pos=player_pos,
            target_rel=target_rel,
            remaining_dist=remaining_dist,
            step_reward=step_reward,
            episode_return=episode_return,
            step_count=step_count,
            critic_value=critic_value,
            action_probs=action_probs,
            chosen_action=chosen_action,
            l_bfs=l_bfs,
            visited_trail=visited_trail,
            bfs_path=bfs_path,
            checkpoint_steps=10000,
            checkpoint_policy_loss=0.045,
            checkpoint_value_loss=0.120,
            checkpoint_success_rate=0.85
        )

        self.assertIsInstance(packed_bytes, bytes)
        self.assertGreater(len(packed_bytes), 200)

        unpacked = InferenceTelemetry.unpack(packed_bytes)
        self.assertIsNotNone(unpacked)
        self.assertEqual(unpacked["step"], step_count)
        self.assertEqual(unpacked["targetPos"], list(target_pos))
        self.assertEqual(unpacked["targetRel"], list(target_rel))
        self.assertAlmostEqual(unpacked["remainingDist"], remaining_dist, places=2)
        self.assertAlmostEqual(unpacked["criticValue"], critic_value, places=2)
        self.assertEqual(unpacked["chosenAction"], chosen_action)
        self.assertEqual(unpacked["lBfs"], l_bfs)
        self.assertEqual(len(unpacked["trailPoints"]), len(visited_trail))
        self.assertEqual(len(unpacked["bfsPoints"]), len(bfs_path))
        self.assertEqual(unpacked["checkpoint"]["steps"], 10000)
        self.assertAlmostEqual(unpacked["checkpoint"]["policyLoss"], 0.045, places=3)

    def test_ipc_emitter_and_reader_roundtrip(self):
        """Verify IPC emitter writes observation + telemetry and reader recovers both intact."""
        emitter = MinecraftIpcEmitter()
        reader = MinecraftIpcReader(emitter.shm_path)

        obs, info = self.env.reset(seed=99)
        tel_bytes = InferenceTelemetry.pack(
            target_pos=self.env.target_pos,
            player_pos=self.env.player_pos,
            target_rel=(3.0, 0.0, 4.0),
            remaining_dist=5.0,
            step_reward=0.2,
            episode_return=1.0,
            step_count=2,
            critic_value=4.5,
            action_probs=[0.8, 0.2],
            chosen_action=0,
            l_bfs=self.env.l_bfs,
            visited_trail=[self.env.player_pos],
            bfs_path=self.env.optimal_path
        )

        full_obs = self.env.extractor.extract_observation(
            player_pos=self.env.player_pos,
            cardinal=self.env.cardinal,
            camera_pitch=self.env.camera_pitch
        )
        emitter.emit_observation(full_obs, game_tick=42, telemetry_bytes=tel_bytes)

        # Read back via MinecraftIpcReader
        header = reader.read_header()
        self.assertIsNotNone(header)
        self.assertEqual(header["gameTick"], 42)

        # Read back telemetry
        telemetry = reader.read_inference_telemetry()
        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry["step"], 2)
        self.assertEqual(telemetry["targetPos"], list(self.env.target_pos))
        self.assertAlmostEqual(telemetry["remainingDist"], 5.0, places=2)
        self.assertAlmostEqual(telemetry["criticValue"], 4.5, places=2)

        emitter.close()
        reader.close()

    def test_single_inference_episode_run(self):
        """Verify run_inference_episode completes and yields valid telemetry metrics."""
        emitter = MinecraftIpcEmitter()
        ckpt_meta = {
            "steps": 5000,
            "latest_policy_loss": 0.03,
            "latest_value_loss": 0.08,
            "latest_success_rate": 0.90
        }

        # Run episode with fps=0 (no sleep delay in tests)
        res = run_inference_episode(
            env=self.env,
            model=None,
            emitter=emitter,
            adapter=self.adapter,
            checkpoint_meta=ckpt_meta,
            fps=0.0,
            greedy=True,
            step_pause=False
        )

        self.assertIn("success", res)
        self.assertIn("steps", res)
        self.assertIn("l_bfs", res)
        self.assertIn("total_return", res)
        self.assertIn("efficiency", res)
        self.assertGreater(res["steps"], 0)

        emitter.close()


if __name__ == "__main__":
    unittest.main()
