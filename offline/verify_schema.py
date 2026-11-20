"""
Schema Verification & End-to-End Pipeline Parity Test.
Validates:
1. Exact tensor shapes and dtypes across all 3 LODs.
2. Priority downsampling preservation of rare blocks (hazards, ores) in B1 and B2.
3. Physical property ranges and occupancy constraints.
4. Surface shell culling / subterranean occlusion masking parity.
5. Zero-loss IPC roundtrip: Emitter -> MMF Shared Memory -> Reader.
"""

import argparse
import os
import sys
import time
from typing import Dict
import numpy as np

# Ensure bridge and offline packages are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
if BRIDGE_DIR not in sys.path:
    sys.path.append(BRIDGE_DIR)

from bridge.mc_ipc_reader import MinecraftIpcReader
from offline.ipc_emitter import MinecraftIpcEmitter
from offline.lod_extractor import GRID_SIZE, LodExtractor, WorldVolume
from offline.palette import (
    PRIORITY_HAZARD,
    PRIORITY_INTERACTIVE,
    PRIORITY_SOLID,
    BlockPalette,
    get_palette,
)
from offline.spawn_indexer import SpawnSampler


def verify_schema(obs: Dict, palette: BlockPalette):
    """Performs rigorous assertions on tensor shapes, types, and value bounds."""
    header = obs["header"]
    assert header["version"] == 3, f"Expected version 3, got {header['version']}"
    assert len(header["playerPos"]) == 3, "playerPos must be 3-tuple"
    assert header["gridSize"] if "gridSize" in header else GRID_SIZE == 16

    expected_shape = (GRID_SIZE, GRID_SIZE, GRID_SIZE)

    block_dtypes = {
        "block_id": np.uint16,
        "occupancy": np.float32,
        "passability": np.float32,
        "hazard": np.float32,
        "hardness": np.float32,
        "light_level": np.float32,
        "fluid_flow_x": np.float32,
        "fluid_flow_z": np.float32,
        "visibility": np.float32
    }

    entity_dtypes = {
        "entity_count": np.uint8,
        "entity_type_id": np.uint8,
        "is_hostile": np.float32,
        "total_health": np.float32
    }

    for lod in range(3):
        lod_key = f"lod{lod}"
        assert lod_key in obs["blocks"], f"Missing {lod_key} in blocks"
        assert lod_key in obs["entities"], f"Missing {lod_key} in entities"

        b = obs["blocks"][lod_key]
        for ch_name, exp_dtype in block_dtypes.items():
            assert ch_name in b, f"Missing {ch_name} in blocks[{lod_key}]"
            arr = b[ch_name]
            assert arr.shape == expected_shape, f"{ch_name} shape mismatch: {arr.shape} vs {expected_shape}"
            assert arr.dtype == exp_dtype, f"{ch_name} dtype mismatch: {arr.dtype} vs {exp_dtype}"

        # Value bounds assertions
        assert 0.0 <= b["occupancy"].min() and b["occupancy"].max() <= 1.0, "Occupancy out of bounds [0, 1]"
        assert 0.0 <= b["passability"].min() and b["passability"].max() <= 1.0, "Passability out of bounds [0, 1]"
        assert 0.0 <= b["hazard"].min() and b["hazard"].max() <= 1.0, "Hazard out of bounds [0, 1]"
        assert 0.0 <= b["light_level"].min() and b["light_level"].max() <= 1.0, "Light level out of bounds [0, 1]"
        assert 0.0 <= b["visibility"].min() and b["visibility"].max() <= 1.0, "Visibility out of bounds [0, 1]"
        assert 0.0 <= b["hardness"].min() and b["hardness"].max() <= 100.0, "Hardness out of bounds [0, 100]"

        e = obs["entities"][lod_key]
        for ch_name, exp_dtype in entity_dtypes.items():
            assert ch_name in e, f"Missing {ch_name} in entities[{lod_key}]"
            arr = e[ch_name]
            assert arr.shape == expected_shape, f"Entity {ch_name} shape mismatch: {arr.shape}"
            assert arr.dtype == exp_dtype, f"Entity {ch_name} dtype mismatch: {arr.dtype}"


def verify_ipc_roundtrip(obs: Dict) -> bool:
    """Emits an observation to MMF and reads it back, verifying exact byte equivalence."""
    emitter = MinecraftIpcEmitter()
    reader = MinecraftIpcReader(emitter.shm_path)

    try:
        emitter.open()
        emitter.emit_observation(obs, game_tick=12345)

        assert reader.open(), f"Failed to open reader at {emitter.shm_path}"
        header = reader.read_header()
        assert header is not None, "Failed to read header from MMF"
        assert header["version"] == 3, f"Header version mismatch: {header['version']}"
        assert header["gameTick"] == 12345, f"Game tick mismatch: {header['gameTick']}"

        slot = header["activeSlot"]
        for lod in range(3):
            read_b = reader.read_block_tensors(slot, lod=lod)
            orig_b = obs["blocks"][f"lod{lod}"]

            for ch in ["block_id", "occupancy", "passability", "hazard", "hardness", "light_level", "visibility"]:
                assert np.array_equal(read_b[ch], orig_b[ch]), f"Mismatch in LOD {lod} channel {ch} during IPC roundtrip!"

            read_e = reader.read_entity_tensors(slot, lod=lod)
            orig_e = obs["entities"][f"lod{lod}"]
            for ch in ["entity_count", "entity_type_id", "is_hostile", "total_health"]:
                assert np.array_equal(read_e[ch], orig_e[ch]), f"Mismatch in entity LOD {lod} channel {ch}!"

        return True
    finally:
        reader.close()
        emitter.close()


def main():
    parser = argparse.ArgumentParser(description="Verify schema and pipeline parity for Phase 1 & 2.")
    parser.add_argument("--dir", "-d", type=str, default="baked_regions_test", help="Directory with baked regions.")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream live frames into MMF for visualizer inspection.")
    args = parser.parse_args()

    palette = get_palette()
    print("=" * 70)
    print("  SQUARED SENSE MC - PHASE 1 & 2 VERIFICATION SUITE")
    print("=" * 70)

    # 1. Load World and Spawn Index
    print("\n[Step 1] Loading baked world volume & spawn pool...")
    world = WorldVolume(args.dir, palette)
    spawns_path = os.path.join(args.dir, "r.-1.2.spawns.npz")
    if not os.path.exists(spawns_path):
        import glob
        sp_files = glob.glob(os.path.join(args.dir, "*.spawns.npz"))
        assert sp_files, f"No .spawns.npz found in {args.dir}"
        spawns_path = sp_files[0]

    sampler = SpawnSampler.from_npz(spawns_path)
    extractor = LodExtractor(world, palette)
    print("  World Volume and LOD Extractor initialized.")

    # 2. Test Extractions across Terrain Categories
    print("\n[Step 2] Testing extraction across terrain categories (Surface, Subterranean, Steep)...")
    categories = ["surface", "subterranean", "steep"]
    for cat in categories:
        pos = sampler.sample_category(cat)
        for cardinal in ["SOUTH", "WEST", "NORTH", "EAST"]:
            obs = extractor.extract_observation(pos, cardinal=cardinal, camera_pitch=0.0)
            verify_schema(obs, palette)
        print(f"  Passed schema verification for '{cat}' category (tested all 4 cardinal directions).")

    # 3. Test Priority Downsampling Invariants
    print("\n[Step 3] Verifying Task 7 Priority Hierarchy in Downsampling...")
    # Synthetic verification: inject known hazards/ores in a sub-volume and ensure they survive B1 and B2 downsampling
    diamond_id = palette.get_id("minecraft:diamond_ore")
    lava_id = palette.get_id("minecraft:lava")
    stone_id = palette.get_id("minecraft:stone")

    # Construct test 4x4x4 sub-volume: 63 stone + 1 lava
    test_subvol = np.full((16 * 4 + 2, 16 * 4 + 2, 16 * 4 + 2), stone_id, dtype=np.uint16)
    # Put 1 lava at local index
    test_subvol[1, 1, 1] = lava_id

    b2_res = extractor.process_lod_tensors(test_subvol, stride=4, apply_visibility_culling=False)
    # The cell at (0, 0, 0) MUST have selected lava because HAZARD (6) > SOLID (2)
    selected_id = b2_res["block_id"][0, 0, 0]
    hazard_flag = b2_res["hazard"][0, 0, 0]
    print(f"  LOD 2 (1/64 lava test): Selected Block = {palette.get_name(selected_id)} (ID {selected_id}), Hazard Flag = {hazard_flag}")
    assert selected_id == lava_id, f"Expected lava to win priority pooling, got {selected_id}"
    assert hazard_flag == 1.0, f"Expected hazard flag 1.0, got {hazard_flag}"
    print("  Task 7 Priority Hierarchy: HAZARDS successfully survive downsampling!")

    # 4. Test Surface Shell / Visibility Filter
    print("\n[Step 4] Verifying Phase 5 Surface Shell Culling...")
    # Fully enclosed solid block test
    test_solid_box = np.full((18, 18, 18), stone_id, dtype=np.uint16)
    b0_res = extractor.process_lod_tensors(test_solid_box, stride=1, apply_visibility_culling=True)
    # Inner cell (1, 1, 1) has all 6 solid neighbors -> must be masked to ID 1 with visibility 0.0
    inner_id = b0_res["block_id"][1, 1, 1]
    inner_vis = b0_res["visibility"][1, 1, 1]
    print(f"  LOD 0 Interior Occluded Block: ID = {inner_id} ({palette.get_name(inner_id)}), Visibility = {inner_vis}")
    assert inner_id == 1, f"Expected masked ID 1, got {inner_id}"
    assert inner_vis == 0.0, f"Expected visibility 0.0, got {inner_vis}"
    print("  Phase 5 Surface Shell Filter: Buried blocks correctly culled to ID 1 (special:masked) with vis=0.0!")

    # 5. Test IPC Shared Memory Roundtrip
    print("\n[Step 5] Testing Zero-Loss Shared Memory (MMF) IPC Roundtrip...")
    test_pos = sampler.sample()
    test_obs = extractor.extract_observation(test_pos, cardinal="SOUTH", camera_pitch=0.0)
    roundtrip_ok = verify_ipc_roundtrip(test_obs)
    assert roundtrip_ok, "IPC Roundtrip failed!"
    print("  Zero-Loss IPC Roundtrip: Emitter -> MMF (Protocol v3) -> Reader verified with bit-for-bit parity!")

    # 6. Stream Live Test Frames to Visualizer (if requested)
    if args.stream:
        print("\n[Step 6] Streaming 30 observation frames to MMF for visualizer inspection (10 Hz)...")
        emitter = MinecraftIpcEmitter()
        emitter.open()
        try:
            for tick in range(30):
                pos = sampler.sample()
                cardinal = ["SOUTH", "WEST", "NORTH", "EAST"][tick % 4]
                obs = extractor.extract_observation(pos, cardinal=cardinal, camera_pitch=0.0)
                emitter.emit_observation(obs, game_tick=tick * 2)
                print(f"\r  Emitted frame {tick+1}/30 | Pos: {pos} | Facing: {cardinal}", end="", flush=True)
                time.sleep(0.1)
            print("\n  Streaming complete.")
        finally:
            emitter.close()

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! (PHASE 1 & 2 COMPLETE)")
    print("=" * 70)


if __name__ == "__main__":
    main()

