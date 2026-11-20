"""
Fast Binary MCA Region Baker.
Converts Minecraft Anvil (.mca) region files into flat, read-only NumPy memory-mapped
arrays (.mmap / .npy) for microsecond zero-copy access during RL training and inference.
"""

import argparse
import glob
import io
import json
import math
import os
import re
import struct
import time
import zlib
from typing import Dict, List, Optional, Tuple
import nbtlib
import numpy as np

from .palette import BlockPalette, get_palette

# Modern Minecraft World Height Bounds (1.18+ / 1.20+ / 1.21)
WORLD_MIN_Y = -64
WORLD_MAX_Y = 319
WORLD_HEIGHT = 384  # 24 vertical sections of 16 blocks each
MIN_SECTION_Y = -4
TOTAL_SECTIONS = 24

REGION_CHUNKS = 32
CHUNK_SIZE = 16
REGION_BLOCKS = REGION_CHUNKS * CHUNK_SIZE  # 512 blocks in X and Z


def parse_region_coords(filename: str) -> Optional[Tuple[int, int]]:
    """Extracts (region_x, region_z) from filename like r.-1.2.mca."""
    base = os.path.basename(filename)
    match = re.match(r"r\.(-?\d+)\.(-?\d+)\.mca$", base)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def unpack_block_states(
    block_states_tag: Dict,
    palette_obj: BlockPalette
) -> np.ndarray:
    """
    Unpacks a 16x16x16 section block_states compound into an array of global token IDs.
    Returns array of shape (16, 16, 16) with axes (X, Y, Z).
    """
    palette_list = block_states_tag.get("palette", [])
    if not palette_list:
        return np.zeros((16, 16, 16), dtype=np.uint16)

    # Map section-local palette to global vocabulary token IDs
    token_ids = np.array(
        [palette_obj.get_id(str(p.get("Name", "minecraft:air"))) for p in palette_list],
        dtype=np.uint16
    )

    if len(palette_list) == 1 or "data" not in block_states_tag:
        # Homogeneous section: entire 16x16x16 is filled with palette[0]
        return np.full((16, 16, 16), token_ids[0], dtype=np.uint16)

    data = block_states_tag["data"]
    bits = max(4, math.ceil(math.log2(len(palette_list))))
    entries_per_long = 64 // bits
    mask = (1 << bits) - 1

    # Vectorized bit-unpacking
    raw_longs = np.array(data, dtype=np.uint64)[:, np.newaxis]
    shifts = np.arange(0, entries_per_long * bits, bits, dtype=np.uint64)[np.newaxis, :]
    unpacked_indices = ((raw_longs >> shifts) & mask).ravel()[:4096]

    # Map to global token IDs
    global_ids = token_ids[unpacked_indices]

    # Minecraft section ordering is (y << 8) | (z << 4) | x -> reshape(16, 16, 16) is (y, z, x)
    # Transpose to (x, y, z): axis 2 (x) -> 0, axis 0 (y) -> 1, axis 1 (z) -> 2
    return global_ids.reshape(16, 16, 16).transpose(2, 0, 1)


def bake_region(
    mca_path: str,
    output_dir: str,
    palette_obj: Optional[BlockPalette] = None,
    overwrite: bool = False
) -> Dict:
    """
    Bakes a single .mca region file into a memory-mapped array of shape (512, 384, 512).
    Saves:
    - r.X.Z.mmap: binary memmap file of uint16 block IDs
    - r.X.Z.heightmap.npy: 2D array (512, 512) of int16 surface world Y
    - r.X.Z.json: metadata manifest
    """
    if palette_obj is None:
        palette_obj = get_palette()

    coords = parse_region_coords(mca_path)
    if coords is None:
        raise ValueError(f"Cannot parse region coordinates from filename: {mca_path}")
    reg_x, reg_z = coords

    os.makedirs(output_dir, exist_ok=True)
    mmap_filename = f"r.{reg_x}.{reg_z}.mmap"
    mmap_path = os.path.join(output_dir, mmap_filename)
    heightmap_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.heightmap.npy")
    json_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.json")

    if not overwrite and os.path.exists(mmap_path) and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    start_time = time.time()
    shape = (REGION_BLOCKS, WORLD_HEIGHT, REGION_BLOCKS)  # (512, 384, 512)

    # Initialize memmap on disk (writable mode)
    # Total size: 512 * 384 * 512 * 2 = 201,326,592 bytes (~192 MB)
    region_blocks = np.memmap(mmap_path, dtype=np.uint16, mode="w+", shape=shape)

    chunks_baked = 0
    with open(mca_path, "rb") as f:
        header = f.read(4096)
        for chunk_idx in range(1024):
            cx = chunk_idx % 32
            cz = chunk_idx // 32
            loc_bytes = header[chunk_idx * 4 : chunk_idx * 4 + 4]
            offset = int.from_bytes(loc_bytes[0:3], "big") * 4096
            sector_count = loc_bytes[3]

            if offset == 0 or sector_count == 0:
                continue

            f.seek(offset)
            length_bytes = f.read(4)
            if len(length_bytes) < 4:
                continue
            chunk_length = int.from_bytes(length_bytes, "big")
            compression = f.read(1)
            compressed_data = f.read(chunk_length - 1)

            try:
                # 32 + zlib.MAX_WBITS automatically detects and decompresses both zlib and gzip
                raw_nbt = zlib.decompress(compressed_data, 32 + zlib.MAX_WBITS)
                nbt = nbtlib.File.parse(io.BytesIO(raw_nbt))
            except Exception:
                continue

            sections = nbt.get("sections", [])
            x_start = cx * 16
            x_end = x_start + 16
            z_start = cz * 16
            z_end = z_start + 16

            for sec in sections:
                sec_y = int(sec.get("Y", -999))
                if sec_y < MIN_SECTION_Y or sec_y >= MIN_SECTION_Y + TOTAL_SECTIONS:
                    continue
                y_idx = sec_y - MIN_SECTION_Y
                y_start = y_idx * 16
                y_end = y_start + 16

                bs = sec.get("block_states")
                if bs:
                    sec_voxels = unpack_block_states(bs, palette_obj)
                    region_blocks[x_start:x_end, y_start:y_end, z_start:z_end] = sec_voxels

            chunks_baked += 1

    # Flush memmap to disk
    region_blocks.flush()

    # Precompute 2D surface heightmap (highest solid non-hazard block world Y at each (x, z))
    solid_mask = palette_obj.lut_is_solid[region_blocks] & (~palette_obj.lut_hazard[region_blocks].astype(bool))
    reversed_solid = solid_mask[:, ::-1, :]
    has_any = np.any(reversed_solid, axis=1)
    highest_reversed_idx = np.argmax(reversed_solid, axis=1)
    highest_y_idx = np.where(has_any, (WORLD_HEIGHT - 1) - highest_reversed_idx, 0)
    world_y_heightmap = (highest_y_idx + WORLD_MIN_Y).astype(np.int16)
    np.save(heightmap_path, world_y_heightmap)

    elapsed = time.time() - start_time
    non_air_count = int(np.count_nonzero(region_blocks != 0))

    # Release memmap file descriptor on Windows
    del region_blocks

    meta = {
        "region_x": reg_x,
        "region_z": reg_z,
        "min_world_x": reg_x * REGION_BLOCKS,
        "max_world_x": reg_x * REGION_BLOCKS + REGION_BLOCKS - 1,
        "min_world_z": reg_z * REGION_BLOCKS,
        "max_world_z": reg_z * REGION_BLOCKS + REGION_BLOCKS - 1,
        "min_world_y": WORLD_MIN_Y,
        "max_world_y": WORLD_MAX_Y,
        "shape": list(shape),
        "dtype": "uint16",
        "chunks_baked": chunks_baked,
        "non_air_blocks": non_air_count,
        "bake_time_seconds": round(elapsed, 2),
        "mmap_file": mmap_filename,
        "heightmap_file": os.path.basename(heightmap_path)
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    return meta


def bake_world(
    world_region_dir: str,
    output_dir: str,
    palette_obj: Optional[BlockPalette] = None,
    overwrite: bool = False
) -> Dict:
    """
    Bakes all .mca region files in a world directory into memmap arrays and writes
    a unified world_manifest.json.
    """
    if palette_obj is None:
        palette_obj = get_palette()

    os.makedirs(output_dir, exist_ok=True)
    mca_files = glob.glob(os.path.join(world_region_dir, "r.*.*.mca"))
    mca_files.sort()

    print(f"[MCABaker] Found {len(mca_files)} region files in {world_region_dir}")
    baked_regions: Dict[str, Dict] = {}
    total_start = time.time()

    min_wx, max_wx = 10**9, -10**9
    min_wz, max_wz = 10**9, -10**9

    for i, mca_path in enumerate(mca_files):
        coords = parse_region_coords(mca_path)
        if not coords:
            continue
        rx, rz = coords
        print(f"[{i+1}/{len(mca_files)}] Baking region r.{rx}.{rz}.mca...", flush=True)
        meta = bake_region(mca_path, output_dir, palette_obj, overwrite=overwrite)
        key = f"{rx},{rz}"
        baked_regions[key] = meta

        min_wx = min(min_wx, meta["min_world_x"])
        max_wx = max(max_wx, meta["max_world_x"])
        min_wz = min(min_wz, meta["min_world_z"])
        max_wz = max(max_wz, meta["max_world_z"])

    total_time = time.time() - total_start
    manifest = {
        "world_region_dir": os.path.abspath(world_region_dir),
        "output_dir": os.path.abspath(output_dir),
        "total_regions": len(baked_regions),
        "world_bounds": {
            "min_x": min_wx if baked_regions else 0,
            "max_x": max_wx if baked_regions else 0,
            "min_z": min_wz if baked_regions else 0,
            "max_z": max_wz if baked_regions else 0,
            "min_y": WORLD_MIN_Y,
            "max_y": WORLD_MAX_Y
        },
        "regions": baked_regions,
        "total_bake_time_seconds": round(total_time, 2)
    }

    manifest_path = os.path.join(output_dir, "world_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"[MCABaker] Finished baking {len(baked_regions)} regions in {total_time:.2f}s.")
    print(f"[MCABaker] Manifest saved to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Bake Minecraft .mca region files into NumPy memmap arrays.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to .mca file or directory containing .mca files.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for baked .mmap and metadata files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already baked files.")

    args = parser.parse_args()
    if os.path.isfile(args.input):
        print(f"Baking single region: {args.input}")
        meta = bake_region(args.input, args.output, overwrite=args.overwrite)
        print(json.dumps(meta, indent=4))
    elif os.path.isdir(args.input):
        bake_world(args.input, args.output, overwrite=args.overwrite)
    else:
        print(f"Error: Input path {args.input} does not exist.")


if __name__ == "__main__":
    main()

