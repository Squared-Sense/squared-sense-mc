"""
Comprehensive Unit & Invariant Tests for Offline Pipeline (Phases 1 & 2).
"""

import os
import unittest
import numpy as np

from offline.palette import (
    BlockPalette,
    get_palette,
    PRIORITY_AIR,
    PRIORITY_FOLIAGE,
    PRIORITY_HAZARD,
    PRIORITY_INTERACTIVE,
    PRIORITY_LIQUID,
    PRIORITY_SOLID,
)
from offline.lod_extractor import (
    calculate_foot_index,
    GRID_SIZE,
    LodExtractor,
    WorldVolume,
)
from offline.spawn_indexer import SpawnSampler, compute_surface_gradient
from offline.ipc_emitter import MinecraftIpcEmitter
from bridge.mc_ipc_reader import MinecraftIpcReader


class TestPaletteInvariants(unittest.TestCase):
    def setUp(self):
        self.palette = get_palette()

    def test_key_block_classifications(self):
        pal = self.palette

        # Grass block must be SOLID, not foliage
        gb_id = pal.get_id("minecraft:grass_block")
        self.assertEqual(pal.lut_priority[gb_id], PRIORITY_SOLID)
        self.assertEqual(pal.lut_passability[gb_id], 0.0)
        self.assertTrue(pal.lut_is_solid[gb_id])
        self.assertTrue(pal.lut_is_opaque[gb_id])

        # Bedrock must be SOLID, not interactive
        br_id = pal.get_id("minecraft:bedrock")
        self.assertEqual(pal.lut_priority[br_id], PRIORITY_SOLID)
        self.assertEqual(pal.lut_passability[br_id], 0.0)
        self.assertEqual(pal.lut_hardness[br_id], 100.0)
        self.assertTrue(pal.lut_is_solid[br_id])

        # Spore blossom must be FOLIAGE, not interactive
        sb_id = pal.get_id("minecraft:spore_blossom")
        self.assertEqual(pal.lut_priority[sb_id], PRIORITY_FOLIAGE)
        self.assertEqual(pal.lut_passability[sb_id], 1.0)
        self.assertFalse(pal.lut_is_solid[sb_id])

        # Heavy core must be SOLID
        hc_id = pal.get_id("minecraft:heavy_core")
        self.assertEqual(pal.lut_priority[hc_id], PRIORITY_SOLID)

        # Lava must be HAZARD
        lava_id = pal.get_id("minecraft:lava")
        self.assertEqual(pal.lut_priority[lava_id], PRIORITY_HAZARD)
        self.assertEqual(pal.lut_hazard[lava_id], 1.0)
        self.assertFalse(pal.lut_is_solid[lava_id])

        # Water must be LIQUID
        water_id = pal.get_id("minecraft:water")
        self.assertEqual(pal.lut_priority[water_id], PRIORITY_LIQUID)
        self.assertEqual(pal.lut_passability[water_id], 0.5)
        self.assertFalse(pal.lut_is_solid[water_id])

        # Air must be AIR
        air_id = pal.get_id("minecraft:air")
        self.assertEqual(air_id, 0)
        self.assertEqual(pal.lut_priority[0], PRIORITY_AIR)
        self.assertEqual(pal.lut_passability[0], 1.0)
        self.assertFalse(pal.lut_is_solid[0])

        # Special masked must be SOLID
        masked_id = pal.get_id("special:masked")
        self.assertEqual(masked_id, 1)
        self.assertEqual(pal.lut_priority[1], PRIORITY_SOLID)
        self.assertEqual(pal.lut_passability[1], 0.0)
        self.assertTrue(pal.lut_is_solid[1])

    def test_get_id_prefix_fallback(self):
        pal = self.palette
        self.assertEqual(pal.get_id("stone"), pal.get_id("minecraft:stone"))
        self.assertEqual(pal.get_id("dirt"), pal.get_id("minecraft:dirt"))
        self.assertEqual(pal.get_id("diamond_ore"), pal.get_id("minecraft:diamond_ore"))

    def test_pitch_window_mapping(self):
        self.assertEqual(calculate_foot_index(0.0), 6)
        self.assertEqual(calculate_foot_index(90.0), 12)
        self.assertEqual(calculate_foot_index(-90.0), 1)
        self.assertEqual(calculate_foot_index(45.0), 9)
        self.assertEqual(calculate_foot_index(-45.0), 4)


class TestLodExtractorInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "baked_regions_test"
        if not os.path.exists(cls.test_dir):
            raise unittest.SkipTest("baked_regions_test directory not found")
        cls.world = WorldVolume(cls.test_dir)
        cls.palette = get_palette()
        cls.extractor = LodExtractor(cls.world, cls.palette)
        sp_path = os.path.join(cls.test_dir, "r.-1.2.spawns.npz")
        cls.sampler = SpawnSampler.from_npz(sp_path)

    def test_multi_cardinal_extractions(self):
        spawn = self.sampler.sample()
        for cardinal in ["SOUTH", "WEST", "NORTH", "EAST"]:
            obs = self.extractor.extract_observation(spawn, cardinal=cardinal, camera_pitch=0.0)
            for lod in range(3):
                b = obs["blocks"][f"lod{lod}"]
                self.assertEqual(b["block_id"].shape, (GRID_SIZE, GRID_SIZE, GRID_SIZE))
                self.assertEqual(b["occupancy"].shape, (GRID_SIZE, GRID_SIZE, GRID_SIZE))
                self.assertTrue(np.all(b["occupancy"] >= 0.0) and np.all(b["occupancy"] <= 1.0))
                self.assertTrue(np.all(b["passability"] >= 0.0) and np.all(b["passability"] <= 1.0))

    def test_unpadded_input_support(self):
        # Directly test process_lod_tensors with an unpadded array
        unpadded = np.zeros((16, 16, 16), dtype=np.uint16)
        res = self.extractor.process_lod_tensors(unpadded, stride=1, apply_visibility_culling=True)
        self.assertEqual(res["block_id"].shape, (16, 16, 16))
        self.assertEqual(res["visibility"].shape, (16, 16, 16))

    def test_spawn_validity(self):
        # Verify that 10 sampled spawns have solid ground and clear air clearance
        for _ in range(10):
            wx, wy, wz = self.sampler.sample()
            floor_id = self.world.get_block(wx, wy - 1, wz)
            feet_id = self.world.get_block(wx, wy, wz)
            head_id = self.world.get_block(wx, wy + 1, wz)

            self.assertTrue(self.palette.lut_is_solid[floor_id], f"Floor {floor_id} not solid at ({wx}, {wy-1}, {wz})")
            self.assertEqual(self.palette.lut_hazard[floor_id], 0.0, f"Floor is hazard at ({wx}, {wy-1}, {wz})")
            self.assertEqual(self.palette.lut_passability[feet_id], 1.0, f"Feet not clear at ({wx}, {wy}, {wz})")
            self.assertEqual(self.palette.lut_passability[head_id], 1.0, f"Head not clear at ({wx}, {wy+1}, {wz})")


class TestIpcEmitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "baked_regions_test"
        if not os.path.exists(cls.test_dir):
            raise unittest.SkipTest("baked_regions_test directory not found")
        cls.world = WorldVolume(cls.test_dir)
        cls.extractor = LodExtractor(cls.world)
        cls.sampler = SpawnSampler.from_npz(os.path.join(cls.test_dir, "r.-1.2.spawns.npz"))

    def test_ping_pong_slot_flip(self):
        emitter = MinecraftIpcEmitter()
        reader = MinecraftIpcReader(emitter.shm_path)
        try:
            emitter.open()
            pos = self.sampler.sample()
            obs = self.extractor.extract_observation(pos, cardinal="SOUTH")

            # Emission 1 -> slot 1
            emitter.emit_observation(obs, game_tick=100)
            self.assertTrue(reader.open())
            h1 = reader.read_header()
            self.assertEqual(h1["gameTick"], 100)
            self.assertEqual(h1["activeSlot"], 1)

            # Emission 2 -> slot 0 (ping-pong flip)
            emitter.emit_observation(obs, game_tick=200)
            h2 = reader.read_header()
            self.assertEqual(h2["gameTick"], 200)
            self.assertEqual(h2["activeSlot"], 0)
        finally:
            reader.close()
            emitter.close()


if __name__ == "__main__":
    unittest.main()

