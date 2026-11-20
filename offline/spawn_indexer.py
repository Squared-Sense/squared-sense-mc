"""
Precomputed Terrain Spawn Indexer.
Scans baked terrain memmaps to index valid spawn coordinates (2-block-tall air clearance
above a solid, non-hazard block), partitioned into the 60/25/15 curriculum split:
- 60% Surface (standard terrain, plains, forests, hills)
- 25% Subterranean (underground cave pockets, Y < 50)
- 15% High-Elevation / Steep (sharp local heightmap gradients, cliffs, ravines)
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .mca_baker import REGION_BLOCKS, WORLD_HEIGHT, WORLD_MIN_Y
from .palette import BlockPalette, get_palette


def compute_surface_gradient(heightmap: np.ndarray) -> np.ndarray:
    """
    Computes max absolute local height difference in 3x3 neighborhood for each cell.
    Returns 2D array of shape (512, 512).
    """
    pad_h = np.pad(heightmap, 1, mode="edge")
    diff_e = np.abs(pad_h[1:-1, 2:] - heightmap)
    diff_w = np.abs(pad_h[1:-1, :-2] - heightmap)
    diff_s = np.abs(pad_h[2:, 1:-1] - heightmap)
    diff_n = np.abs(pad_h[:-2, 1:-1] - heightmap)
    diff_se = np.abs(pad_h[2:, 2:] - heightmap)
    diff_nw = np.abs(pad_h[:-2, :-2] - heightmap)
    diff_sw = np.abs(pad_h[2:, :-2] - heightmap)
    diff_ne = np.abs(pad_h[:-2, 2:] - heightmap)

    max_grad = np.maximum.reduce([
        diff_e, diff_w, diff_s, diff_n,
        diff_se, diff_nw, diff_sw, diff_ne
    ])
    return max_grad


def index_region_spawns(
    mmap_path: str,
    heightmap_path: str,
    meta: Dict,
    palette_obj: Optional[BlockPalette] = None,
    steep_threshold: int = 3,
    subterranean_max_y: int = 50
) -> Dict[str, np.ndarray]:
    """
    Scans a single baked region memmap and returns categorized spawn arrays:
    - 'surface': shape (N_surf, 3)
    - 'subterranean': shape (N_sub, 3)
    - 'steep': shape (N_steep, 3)
    Coordinates are in absolute Minecraft world space (X, Y, Z).
    """
    if palette_obj is None:
        palette_obj = get_palette()

    min_wx = meta["min_world_x"]
    min_wz = meta["min_world_z"]

    mmap = np.memmap(mmap_path, dtype=np.uint16, mode="r", shape=(REGION_BLOCKS, WORLD_HEIGHT, REGION_BLOCKS))
    heightmap = np.load(heightmap_path)  # (512, 512) int16

    # 1. Gradient & Steep Surface Mask
    grad = compute_surface_gradient(heightmap)
    is_steep_column = (grad >= steep_threshold)

    # 2. Surface Spawns Scan
    surface_spawns_list: List[np.ndarray] = []
    steep_spawns_list: List[np.ndarray] = []

    # Vectorized check along the 2D surface
    rx_indices, rz_indices = np.meshgrid(np.arange(REGION_BLOCKS), np.arange(REGION_BLOCKS), indexing="ij")
    rx_flat = rx_indices.ravel()
    rz_flat = rz_indices.ravel()
    surf_wy = heightmap[rx_flat, rz_flat]
    surf_y_idx = surf_wy - WORLD_MIN_Y

    # Ensure valid vertical bounds for floor, feet, head
    valid_bounds = (surf_y_idx >= 0) & (surf_y_idx + 2 < WORLD_HEIGHT)
    rx_valid = rx_flat[valid_bounds]
    rz_valid = rz_flat[valid_bounds]
    y_floor_idx = surf_y_idx[valid_bounds]
    y_feet_idx = y_floor_idx + 1
    y_head_idx = y_floor_idx + 2

    # Query blocks directly
    floor_blocks = mmap[rx_valid, y_floor_idx, rz_valid]
    feet_blocks = mmap[rx_valid, y_feet_idx, rz_valid]
    head_blocks = mmap[rx_valid, y_head_idx, rz_valid]

    floor_ok = palette_obj.lut_is_solid[floor_blocks] & (palette_obj.lut_hazard[floor_blocks] == 0)
    feet_ok = (palette_obj.lut_passability[feet_blocks] == 1.0) & (palette_obj.lut_hazard[feet_blocks] == 0)
    head_ok = (palette_obj.lut_passability[head_blocks] == 1.0) & (palette_obj.lut_hazard[head_blocks] == 0)

    valid_surface = floor_ok & feet_ok & head_ok

    surf_wx = rx_valid[valid_surface] + min_wx
    surf_wz = rz_valid[valid_surface] + min_wz
    surf_wy_valid = y_feet_idx[valid_surface] + WORLD_MIN_Y

    steep_flags = is_steep_column[rx_valid[valid_surface], rz_valid[valid_surface]]

    # Steep coordinates
    steep_coords = np.column_stack([
        surf_wx[steep_flags],
        surf_wy_valid[steep_flags],
        surf_wz[steep_flags]
    ]).astype(np.int32)

    # Standard surface coordinates (non-steep)
    standard_surf_coords = np.column_stack([
        surf_wx[~steep_flags],
        surf_wy_valid[~steep_flags],
        surf_wz[~steep_flags]
    ]).astype(np.int32)

    # 3. Subterranean Spawns Scan (Caves & underground tunnels, Y < 50)
    subterranean_coords_list: List[np.ndarray] = []
    max_sub_y_idx = min(WORLD_HEIGHT - 2, subterranean_max_y - WORLD_MIN_Y)

    for y_idx in range(1, max_sub_y_idx + 1):
        wy = y_idx + WORLD_MIN_Y
        floor = mmap[:, y_idx - 1, :]
        feet = mmap[:, y_idx, :]
        head = mmap[:, y_idx + 1, :]

        floor_ok = palette_obj.lut_is_solid[floor] & (palette_obj.lut_hazard[floor] == 0)
        feet_ok = (palette_obj.lut_passability[feet] == 1.0) & (palette_obj.lut_hazard[feet] == 0)
        head_ok = (palette_obj.lut_passability[head] == 1.0) & (palette_obj.lut_hazard[head] == 0)

        # Must be underground (below the surface heightmap)
        underground = (wy < heightmap)

        valid_sub = floor_ok & feet_ok & head_ok & underground
        valid_indices = np.argwhere(valid_sub)
        if len(valid_indices) > 0:
            sub_wx = valid_indices[:, 0] + min_wx
            sub_wz = valid_indices[:, 1] + min_wz
            sub_wy = np.full(len(valid_indices), wy, dtype=np.int32)
            coords_3d = np.column_stack([sub_wx, sub_wy, sub_wz]).astype(np.int32)
            subterranean_coords_list.append(coords_3d)

    subterranean_coords = (
        np.vstack(subterranean_coords_list)
        if subterranean_coords_list
        else np.zeros((0, 3), dtype=np.int32)
    )

    return {
        "surface": standard_surf_coords,
        "subterranean": subterranean_coords,
        "steep": steep_coords
    }


def index_and_save_region_spawns(
    output_dir: str,
    meta: Dict,
    palette_obj: Optional[BlockPalette] = None
) -> str:
    """Computes and saves spawns for a single region as an .npz file."""
    reg_x = meta["region_x"]
    reg_z = meta["region_z"]
    mmap_path = os.path.join(output_dir, meta["mmap_file"])
    heightmap_path = os.path.join(output_dir, meta["heightmap_file"])
    npz_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.spawns.npz")
    summary_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.spawns.json")

    start = time.time()
    pools = index_region_spawns(mmap_path, heightmap_path, meta, palette_obj)
    np.savez_compressed(
        npz_path,
        surface=pools["surface"],
        subterranean=pools["subterranean"],
        steep=pools["steep"]
    )

    elapsed = time.time() - start
    summary = {
        "region_x": reg_x,
        "region_z": reg_z,
        "surface_count": len(pools["surface"]),
        "subterranean_count": len(pools["subterranean"]),
        "steep_count": len(pools["steep"]),
        "total_spawns": len(pools["surface"]) + len(pools["subterranean"]) + len(pools["steep"]),
        "elapsed_seconds": round(elapsed, 2),
        "spawns_file": os.path.basename(npz_path)
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return npz_path


class SpawnSampler:
    """
    Curriculum-weighted spawn sampler providing fast random spawn coordinates
    split by 60% Surface, 25% Subterranean, 15% Steep.
    """

    def __init__(
        self,
        surface_spawns: np.ndarray,
        subterranean_spawns: np.ndarray,
        steep_spawns: np.ndarray,
        weights: Tuple[float, float, float] = (0.60, 0.25, 0.15)
    ):
        self.surface = surface_spawns
        self.subterranean = subterranean_spawns
        self.steep = steep_spawns
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()

        self.categories = ["surface", "subterranean", "steep"]
        self.pools = [self.surface, self.subterranean, self.steep]

        print(
            f"[SpawnSampler] Initialized with {len(self.surface):,} surface, "
            f"{len(self.subterranean):,} subterranean, {len(self.steep):,} steep spawns."
        )

    @classmethod
    def from_npz(cls, npz_path: str, weights: Tuple[float, float, float] = (0.60, 0.25, 0.15)) -> "SpawnSampler":
        data = np.load(npz_path)
        return cls(
            surface_spawns=data["surface"],
            subterranean_spawns=data["subterranean"],
            steep_spawns=data["steep"],
            weights=weights
        )

    def sample(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int, int]:
        """Samples a single (wx, wy, wz) spawn point according to curriculum weights."""
        if rng is None:
            rng = np.random.default_rng()

        cat_idx = rng.choice(3, p=self.weights)
        pool = self.pools[cat_idx]

        if len(pool) == 0:
            # Fallback to any non-empty pool
            for p in self.pools:
                if len(p) > 0:
                    pool = p
                    break

        if len(pool) == 0:
            raise RuntimeError("All spawn pools are empty!")

        idx = rng.integers(0, len(pool))
        coord = pool[idx]
        return int(coord[0]), int(coord[1]), int(coord[2])

    def sample_category(self, category: str, rng: Optional[np.random.Generator] = None) -> Tuple[int, int, int]:
        """Samples a spawn strictly from the requested terrain category."""
        if rng is None:
            rng = np.random.default_rng()

        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. Choose from {self.categories}")

        pool = self.pools[self.categories.index(category)]
        if len(pool) == 0:
            raise RuntimeError(f"Spawn pool for category '{category}' is empty!")

        idx = rng.integers(0, len(pool))
        coord = pool[idx]
        return int(coord[0]), int(coord[1]), int(coord[2])


def main():
    parser = argparse.ArgumentParser(description="Precompute terrain spawn index from baked memmap.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory containing baked .mmap and .json files.")
    parser.add_argument("--region", "-r", type=str, default=None, help="Optional specific region like r.-1.2.")

    args = parser.parse_args()
    palette = get_palette()

    if args.region:
        json_path = os.path.join(args.dir, f"{args.region}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Indexing spawns for region {args.region}...")
        npz = index_and_save_region_spawns(args.dir, meta, palette)
        print(f"Saved to {npz}")
    else:
        all_json = glob.glob(os.path.join(args.dir, "r.*.*.json"))
        # Exclude summary files like r.X.Z.spawns.json
        json_files = [f for f in all_json if not f.endswith(".spawns.json")]
        print(f"Found {len(json_files)} region manifests in {args.dir}")
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rx, rz = meta["region_x"], meta["region_z"]
            print(f"Indexing spawns for r.{rx}.{rz}...", flush=True)
            index_and_save_region_spawns(args.dir, meta, palette)
        print("Done indexing all regions.")


if __name__ == "__main__":
    main()

