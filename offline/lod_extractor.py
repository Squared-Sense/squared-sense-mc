"""
NumPy LOD Extractor (Unified Schema).
Slices and downsamples B0, B1, B2 observation tensors on the fly directly from
baked memory-mapped Minecraft regions with sub-millisecond latency.
Matches exact schema expected by RL models, bridge, and 3D visualizer.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .mca_baker import REGION_BLOCKS, WORLD_HEIGHT, WORLD_MIN_Y
from .palette import BlockPalette, get_palette

GRID_SIZE = 16
PLAYER_FORWARD_INDEX = 2
PLAYER_LATERAL_INDEX = 8
DEFAULT_FOOT_INDEX = 6
LOD_STRIDES = (1, 2, 4)

CARDINAL_DIRECTIONS = {
    "SOUTH": {"yaw": 0.0,   "forward": (0, 1),  "right": (-1, 0)},
    "WEST":  {"yaw": 90.0,  "forward": (-1, 0), "right": (0, -1)},
    "NORTH": {"yaw": 180.0, "forward": (0, -1), "right": (1, 0)},
    "EAST":  {"yaw": 270.0, "forward": (1, 0),  "right": (0, 1)}
}


def calculate_foot_index(pitch: float) -> int:
    """
    Calculates foot index gy in [1..12] from camera pitch in [-90, +90],
    matching org.sqsense.squared_sense_mc.client.observation.geometry.PitchWindowCalculator.
    """
    clamped_pitch = max(-90.0, min(90.0, pitch))
    if clamped_pitch >= 0.0:
        # Looking down: foot index shifts from 6 up to 12
        return int(round(6 + (clamped_pitch / 90.0) * 6))
    else:
        # Looking up: foot index shifts from 6 down to 1
        return int(round(6 - (-clamped_pitch / 90.0) * 5))


class WorldVolume:
    """
    Manages one or more baked region memmaps, allowing seamless world-space slicing.
    """

    def __init__(self, region_dir: str, palette_obj: Optional[BlockPalette] = None):
        self.region_dir = region_dir
        self.palette = palette_obj or get_palette()
        self.regions: Dict[Tuple[int, int], np.ndarray] = {}
        self.manifest: Dict = {}

        manifest_path = os.path.join(region_dir, "world_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

        self._discover_regions()

    def _discover_regions(self):
        """Discovers and maps all r.X.Z.mmap files in region_dir."""
        import glob
        import re

        mmap_files = glob.glob(os.path.join(self.region_dir, "r.*.*.mmap"))
        for mf in mmap_files:
            base = os.path.basename(mf)
            m = re.match(r"r\.(-?\d+)\.(-?\d+)\.mmap$", base)
            if m:
                rx, rz = int(m.group(1)), int(m.group(2))
                mmap_arr = np.memmap(
                    mf,
                    dtype=np.uint16,
                    mode="r",
                    shape=(REGION_BLOCKS, WORLD_HEIGHT, REGION_BLOCKS)
                )
                self.regions[(rx, rz)] = mmap_arr

        print(f"[WorldVolume] Loaded {len(self.regions)} region memmaps from {self.region_dir}")

    def get_block(self, wx: int, wy: int, wz: int) -> int:
        """Returns the integer block token ID at world coordinate (wx, wy, wz)."""
        y_idx = wy - WORLD_MIN_Y
        if y_idx < 0 or y_idx >= WORLD_HEIGHT:
            return 0  # air

        rx = wx >> 9  # wx // 512
        rz = wz >> 9  # wz // 512
        mmap = self.regions.get((rx, rz))
        if mmap is None:
            return 0  # air

        bx = wx & 511  # wx % 512
        bz = wz & 511  # wz % 512
        return int(mmap[bx, y_idx, bz])

    def query_coordinates(self, wx_arr: np.ndarray, wy_arr: np.ndarray, wz_arr: np.ndarray) -> np.ndarray:
        """
        Queries block IDs for 3D coordinate arrays of identical shape (X, Y, Z).
        Optimized for within-region direct queries with fallback for boundary crossing.
        """
        shape = wx_arr.shape
        out = np.zeros(shape, dtype=np.uint16)

        y_idx = wy_arr - WORLD_MIN_Y
        valid_y = (y_idx >= 0) & (y_idx < WORLD_HEIGHT)

        min_rx, max_rx = int(np.min(wx_arr) >> 9), int(np.max(wx_arr) >> 9)
        min_rz, max_rz = int(np.min(wz_arr) >> 9), int(np.max(wz_arr) >> 9)

        rx_arr = wx_arr >> 9
        rz_arr = wz_arr >> 9

        for rx in range(min_rx, max_rx + 1):
            for rz in range(min_rz, max_rz + 1):
                mmap = self.regions.get((rx, rz))
                if mmap is None:
                    continue

                mask = (rx_arr == rx) & (rz_arr == rz) & valid_y
                if not np.any(mask):
                    continue

                bx = wx_arr[mask] & 511
                by = y_idx[mask]
                bz = wz_arr[mask] & 511
                out[mask] = mmap[bx, by, bz]

        return out


class LodExtractor:
    """
    Extracts multi-LOD observation tensors (B0, B1, B2) and entity grids from a WorldVolume.
    """

    def __init__(self, world_volume: WorldVolume, palette_obj: Optional[BlockPalette] = None):
        self.world = world_volume
        self.palette = palette_obj or self.world.palette

        # Precompute coordinate grids for each LOD
        self._precompute_relative_grids()

    def _precompute_relative_grids(self):
        """Precomputes relative local index offsets for strides 1, 2, 4."""
        self.local_offsets: Dict[int, Dict[str, np.ndarray]] = {}

        for stride in LOD_STRIDES:
            total_span = GRID_SIZE * stride  # 16, 32, 64
            # lx: (gx - 8) * stride + ox
            # For each gx in 0..15, ox in 0..stride-1
            gx_indices = np.arange(GRID_SIZE)
            ox_indices = np.arange(stride)
            lx = ((gx_indices[:, np.newaxis] - PLAYER_LATERAL_INDEX) * stride + ox_indices[np.newaxis, :]).ravel()

            # gz: (gz - 2) * stride + oz
            gz_indices = np.arange(GRID_SIZE)
            oz_indices = np.arange(stride)
            lz = ((gz_indices[:, np.newaxis] - PLAYER_FORWARD_INDEX) * stride + oz_indices[np.newaxis, :]).ravel()

            self.local_offsets[stride] = {
                "lx": lx,
                "lz": lz,
                "total_span": total_span
            }

    def _get_ly(self, stride: int, foot_index: int) -> np.ndarray:
        gy_indices = np.arange(GRID_SIZE)
        oy_indices = np.arange(stride)
        return ((gy_indices[:, np.newaxis] - foot_index) * stride + oy_indices[np.newaxis, :]).ravel()

    def extract_lod_raw_voxels(
        self,
        player_pos: Tuple[int, int, int],
        cardinal: str,
        foot_index: int,
        stride: int,
        pad: int = 0
    ) -> np.ndarray:
        """
        Extracts 3D raw voxel IDs covering the entire volume of a given stride.
        Shape: (16*stride + 2*pad, 16*stride + 2*pad, 16*stride + 2*pad)
        with local axes (X_local, Y_local, Z_local).
        Uses ultra-fast direct 3D memmap box slicing (sub-millisecond).
        """
        px, py, pz = player_pos
        offsets = self.local_offsets[stride]
        lx = offsets["lx"]
        lz = offsets["lz"]
        ly = self._get_ly(stride, foot_index)

        if pad > 0:
            lx_ext = np.concatenate([lx[0] - np.arange(pad, 0, -1), lx, lx[-1] + np.arange(1, pad + 1)])
            ly_ext = np.concatenate([ly[0] - np.arange(pad, 0, -1), ly, ly[-1] + np.arange(1, pad + 1)])
            lz_ext = np.concatenate([lz[0] - np.arange(pad, 0, -1), lz, lz[-1] + np.arange(1, pad + 1)])
        else:
            lx_ext, ly_ext, lz_ext = lx, ly, lz

        min_lx, max_lx = int(lx_ext[0]), int(lx_ext[-1])
        min_ly, max_ly = int(ly_ext[0]), int(ly_ext[-1])
        min_lz, max_lz = int(lz_ext[0]), int(lz_ext[-1])

        # Compute bounding intervals in world space
        if cardinal == "SOUTH":
            min_wx, max_wx = px - max_lx, px - min_lx
            min_wy, max_wy = py + min_ly, py + max_ly
            min_wz, max_wz = pz + min_lz, pz + max_lz
        elif cardinal == "NORTH":
            min_wx, max_wx = px + min_lx, px + max_lx
            min_wy, max_wy = py + min_ly, py + max_ly
            min_wz, max_wz = pz - max_lz, pz - min_lz
        elif cardinal == "WEST":
            min_wx, max_wx = px - max_lz, px - min_lz
            min_wy, max_wy = py + min_ly, py + max_ly
            min_wz, max_wz = pz - max_lx, pz - min_lx
        else:  # EAST
            min_wx, max_wx = px + min_lz, px + max_lz
            min_wy, max_wy = py + min_ly, py + max_ly
            min_wz, max_wz = pz + min_lx, pz + max_lx

        min_rx, max_rx = min_wx >> 9, max_wx >> 9
        min_rz, max_rz = min_wz >> 9, max_wz >> 9
        min_by = min_wy - WORLD_MIN_Y
        max_by = max_wy - WORLD_MIN_Y

        # Ultra-fast single-region path
        if min_rx == max_rx and min_rz == max_rz and min_by >= 0 and max_by < WORLD_HEIGHT:
            mmap = self.world.regions.get((min_rx, min_rz))
            if mmap is not None:
                bx0 = min_wx & 511
                bx1 = (max_wx & 511) + 1
                bz0 = min_wz & 511
                bz1 = (max_wz & 511) + 1
                by0 = min_by
                by1 = max_by + 1

                box = mmap[bx0:bx1, by0:by1, bz0:bz1]

                if cardinal == "SOUTH":
                    return box[::-1, :, :]
                elif cardinal == "NORTH":
                    return box[:, :, ::-1]
                elif cardinal == "WEST":
                    return box.transpose(2, 1, 0)[::-1, :, ::-1]
                else:  # EAST
                    return box.transpose(2, 1, 0)[:, :, :]

        # Multi-region / boundary fallback
        if cardinal == "SOUTH":
            wx_1d = px - lx_ext
            wz_1d = pz + lz_ext
            wx_3d, wy_3d, wz_3d = np.meshgrid(wx_1d, py + ly_ext, wz_1d, indexing="ij")
        elif cardinal == "NORTH":
            wx_1d = px + lx_ext
            wz_1d = pz - lz_ext
            wx_3d, wy_3d, wz_3d = np.meshgrid(wx_1d, py + ly_ext, wz_1d, indexing="ij")
        elif cardinal == "WEST":
            wx_1d = px - lz_ext
            wz_1d = pz - lx_ext
            wz_3d, wy_3d, wx_3d = np.meshgrid(wz_1d, py + ly_ext, wx_1d, indexing="ij")
        else:  # EAST
            wx_1d = px + lz_ext
            wz_1d = pz + lx_ext
            wz_3d, wy_3d, wx_3d = np.meshgrid(wz_1d, py + ly_ext, wx_1d, indexing="ij")

        return self.world.query_coordinates(wx_3d, wy_3d, wz_3d)

    def process_lod_tensors(
        self,
        raw_voxels_padded: np.ndarray,
        stride: int,
        apply_visibility_culling: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Downsamples raw voxels into the standardized 16x16x16 block channel tensors:
        - block_id: uint16
        - occupancy: float32
        - passability: float32
        - hazard: float32
        - hardness: float32
        - light_level: float32
        - fluid_flow_x: float32
        - fluid_flow_z: float32
        - visibility: float32
        """
        # Handle padded vs unpadded raw voxels
        expected_padded_dim = GRID_SIZE * stride + 2
        if raw_voxels_padded.shape[0] == expected_padded_dim:
            voxels = raw_voxels_padded[1:-1, 1:-1, 1:-1]
            padded_voxels = raw_voxels_padded
        else:
            voxels = raw_voxels_padded
            padded_voxels = np.pad(voxels, 1, mode="edge")

        grid_shape = (GRID_SIZE, GRID_SIZE, GRID_SIZE)

        if stride == 1:
            # Fast LOD 0 Path (1:1 cell mapping)
            block_id = voxels.copy()
            occupancy = self.palette.lut_is_solid[block_id].astype(np.float32)
            passability = self.palette.lut_passability[block_id]
            hazard = self.palette.lut_hazard[block_id]
            hardness = self.palette.lut_hardness[block_id]
            light_level = (self.palette.lut_luminance[block_id] / 15.0).astype(np.float32)
            fluid_flow_x = np.zeros(grid_shape, dtype=np.float32)
            fluid_flow_z = np.zeros(grid_shape, dtype=np.float32)
            visibility = np.ones(grid_shape, dtype=np.float32)

            if apply_visibility_culling:
                # 6-face occlusion test: check if all 6 neighbor faces are opaque cubes
                is_opaque = self.palette.lut_is_opaque[padded_voxels]
                up = is_opaque[1:-1, 2:, 1:-1]
                down = is_opaque[1:-1, :-2, 1:-1]
                left = is_opaque[:-2, 1:-1, 1:-1]
                right = is_opaque[2:, 1:-1, 1:-1]
                front = is_opaque[1:-1, 1:-1, 2:]
                back = is_opaque[1:-1, 1:-1, :-2]

                fully_enclosed = up & down & left & right & front & back
                occluded = (occupancy == 1.0) & fully_enclosed
                visibility[occluded] = 0.0
                block_id[occluded] = 1  # special:masked

            return {
                "block_id": block_id,
                "occupancy": occupancy,
                "passability": passability,
                "hazard": hazard,
                "hardness": hardness,
                "light_level": light_level,
                "fluid_flow_x": fluid_flow_x,
                "fluid_flow_z": fluid_flow_z,
                "visibility": visibility
            }

        # LOD 1 (stride 2, 8 sub-blocks) or LOD 2 (stride 4, 64 sub-blocks)
        sub_count = stride * stride * stride
        # Reshape to (16, stride, 16, stride, 16, stride) then transpose to (16, 16, 16, sub_count)
        grouped = voxels.reshape(
            GRID_SIZE, stride,
            GRID_SIZE, stride,
            GRID_SIZE, stride
        ).transpose(0, 2, 4, 1, 3, 5).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, sub_count)

        # 1. Occupancy: ratio of solid blocks
        occupancy = np.mean(self.palette.lut_is_solid[grouped], axis=-1, dtype=np.float32)

        # 2. Passability: average passability
        passability = np.mean(self.palette.lut_passability[grouped], axis=-1, dtype=np.float32)

        # 3. Hazard: 1.0 if ANY sub-block is hazard
        hazard = (np.sum(self.palette.lut_hazard[grouped], axis=-1) > 0).astype(np.float32)

        # 4. Hardness: average hardness
        hardness = np.mean(self.palette.lut_hardness[grouped], axis=-1, dtype=np.float32)

        # 5. Light level: average normalized luminance
        light_level = np.mean(self.palette.lut_luminance[grouped] / 15.0, axis=-1, dtype=np.float32)

        # 6. Fluid flow (zero-filled for static terrain)
        fluid_flow_x = np.zeros(grid_shape, dtype=np.float32)
        fluid_flow_z = np.zeros(grid_shape, dtype=np.float32)

        # 7. Priority Downsampling for block IDs
        prios = self.palette.lut_priority[grouped]  # (16, 16, 16, sub_count)
        max_prio = np.max(prios, axis=-1, keepdims=True)
        is_max = (prios == max_prio)

        g_flat = grouped.reshape(GRID_SIZE * GRID_SIZE * GRID_SIZE, sub_count)
        m_flat = is_max.reshape(GRID_SIZE * GRID_SIZE * GRID_SIZE, sub_count)
        block_id_flat = np.zeros(GRID_SIZE * GRID_SIZE * GRID_SIZE, dtype=np.uint16)

        # Check homogeneous cells first (very fast path for air and bulk solid rock)
        homo = (g_flat[:, 0:1] == g_flat).all(axis=1)
        block_id_flat[homo] = g_flat[homo, 0]

        # For mixed cells: select mode among highest-priority sub-blocks
        mixed_indices = np.where(~homo)[0]
        for idx in mixed_indices:
            row = g_flat[idx]
            m_row = m_flat[idx]
            valid_blocks = row[m_row]
            counts = np.bincount(valid_blocks)
            block_id_flat[idx] = np.argmax(counts)

        block_id = block_id_flat.reshape(grid_shape)
        visibility = np.ones(grid_shape, dtype=np.float32)

        if apply_visibility_culling:
            # Sub-volume occlusion: if cell is 100% solid and all outer faces are opaque
            # Check outer faces in raw_voxels_padded
            # For fast vectorization: cells with occupancy < 1.0 are definitely exposed
            # For occupancy == 1.0: test if enclosed by solid neighbors
            is_opaque_sub = self.palette.lut_is_opaque[grouped].all(axis=-1)
            # Pad is_opaque_sub to test neighbors
            padded_sub = np.pad(is_opaque_sub, 1, mode="constant", constant_values=False)
            up = padded_sub[1:-1, 2:, 1:-1]
            down = padded_sub[1:-1, :-2, 1:-1]
            left = padded_sub[:-2, 1:-1, 1:-1]
            right = padded_sub[2:, 1:-1, 1:-1]
            front = padded_sub[1:-1, 1:-1, 2:]
            back = padded_sub[1:-1, 1:-1, :-2]
            occluded = (occupancy == 1.0) & up & down & left & right & front & back
            visibility[occluded] = 0.0
            block_id[occluded] = 1  # special:masked

        return {
            "block_id": block_id,
            "occupancy": occupancy,
            "passability": passability,
            "hazard": hazard,
            "hardness": hardness,
            "light_level": light_level,
            "fluid_flow_x": fluid_flow_x,
            "fluid_flow_z": fluid_flow_z,
            "visibility": visibility
        }

    def extract_observation(
        self,
        player_pos: Tuple[int, int, int],
        cardinal: str = "SOUTH",
        camera_pitch: float = 0.0,
        apply_visibility_culling: bool = True
    ) -> Dict:
        """
        Extracts full multi-LOD observation dictionary (B0, B1, B2 + zeroed entity grids)
        in the exact schema of the live mod and visualizer.
        """
        foot_index = calculate_foot_index(camera_pitch)
        grid_shape = (GRID_SIZE, GRID_SIZE, GRID_SIZE)

        blocks: Dict[str, Dict[str, np.ndarray]] = {}
        for lod, stride in enumerate(LOD_STRIDES):
            raw_padded = self.extract_lod_raw_voxels(
                player_pos=player_pos,
                cardinal=cardinal,
                foot_index=foot_index,
                stride=stride,
                pad=1
            )
            blocks[f"lod{lod}"] = self.process_lod_tensors(
                raw_voxels_padded=raw_padded,
                stride=stride,
                apply_visibility_culling=apply_visibility_culling
            )

        # Static placeholders for dynamic entity tensors
        entities: Dict[str, Dict[str, np.ndarray]] = {}
        for lod in range(3):
            entities[f"lod{lod}"] = {
                "entity_count": np.zeros(grid_shape, dtype=np.uint8),
                "entity_type_id": np.zeros(grid_shape, dtype=np.uint8),
                "is_hostile": np.zeros(grid_shape, dtype=np.float32),
                "total_health": np.zeros(grid_shape, dtype=np.float32)
            }

        header = {
            "version": 3,
            "playerPos": (float(player_pos[0]), float(player_pos[1]), float(player_pos[2])),
            "cameraYaw": CARDINAL_DIRECTIONS[cardinal]["yaw"],
            "cameraPitch": camera_pitch,
            "snappedYaw": CARDINAL_DIRECTIONS[cardinal]["yaw"],
            "cardinal": cardinal,
            "footIndex": foot_index,
            "forwardIndex": PLAYER_FORWARD_INDEX,
            "lateralIndex": PLAYER_LATERAL_INDEX,
            "vocabSize": self.palette.vocab_size,
            "strides": LOD_STRIDES,
            "visibilityFilter": apply_visibility_culling
        }

        return {
            "header": header,
            "blocks": blocks,
            "entities": entities
        }
