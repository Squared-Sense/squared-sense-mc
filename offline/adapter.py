"""
Observation & State Adapter for Squared Sense MC and World Model Interoperability.

Transforms raw environment observations (LOD categorical block tokens and physical channels)
into the strict 16-channel continuous volumetric tensor representation required by the
companion World Model (docs/WORLD_MODEL_CONTRACTS.md):
  - shells: (3, 16, 16, 16, 16) float32 in [0.0, 1.0] in channel-last layout (LOD, X, Y, Z, C)
  - state: (12,) float32 proprioception
  - goal_vec: (3,) float32 normalized 3D relative goal direction
  - meta: (6,) float32 world and relative goal coordinates
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from offline.palette import BlockPalette

CARDINAL_ORDER = ["SOUTH", "WEST", "NORTH", "EAST"]

CARDINAL_VECTORS = {
    "SOUTH": {"forward": (0, 1),  "right": (-1, 0)},
    "WEST":  {"forward": (-1, 0), "right": (0, -1)},
    "NORTH": {"forward": (0, -1), "right": (1, 0)},
    "EAST":  {"forward": (1, 0),  "right": (0, 1)}
}

# 16-Channel Schema Constants matching docs/WORLD_MODEL_CONTRACTS.md & data/schema.py
CH_TERRAIN = 0
CH_AIR = 1
CH_HAZARD = 2
CH_LIQUID = 3
CH_CLIMBABLE = 4
CH_FOLIAGE = 5
CH_WORLD_Y = 6
CH_COLLISION_HEIGHT = 7
CH_WAYPOINT = 8
CH_RESOURCE = 9
CH_WALKABLE_SURFACE = 10
CH_LIGHT_LEVEL = 11
CH_LETHAL_DROP = 12
CH_SURFACE_SHELL = 13
CH_BLOCK_HARDNESS = 14
CH_OUT_OF_BOUNDS = 15

NUM_CHANNELS = 16
NUM_LODS = 3
GRID_SIZE = 16

CARDINAL_YAW_DEG = {
    "SOUTH": 0.0,
    "WEST": 90.0,
    "NORTH": 180.0,
    "EAST": 270.0
}


class ObservationAdapter:
    """
    Vectorized high-throughput adapter converting environment observations into
    World Model compatible 16-channel float32 tensors and proprioceptive vectors.
    """

    def __init__(self, palette: Optional[BlockPalette] = None):
        self.palette = palette or BlockPalette()
        self._build_category_luts()

    def _build_category_luts(self):
        """Precomputes fast Boolean/float LUTs across the 1,059 vocabulary for 16 channels."""
        vocab_size = self.palette.vocab_size

        self.is_liquid_lut = np.zeros(vocab_size, dtype=np.bool_)
        self.is_climbable_lut = np.zeros(vocab_size, dtype=np.bool_)
        self.is_foliage_lut = np.zeros(vocab_size, dtype=np.bool_)
        self.is_resource_lut = np.zeros(vocab_size, dtype=np.bool_)
        self.collision_height_lut = np.zeros(vocab_size, dtype=np.float32)

        for name, token_id in self.palette.block_to_id.items():
            if token_id >= vocab_size:
                continue

            # Liquid
            if "water" in name or "bubble_column" in name:
                self.is_liquid_lut[token_id] = True

            # Climbable / Vertical step
            if "ladder" in name or "vine" in name or "scaffolding" in name or "stairs" in name:
                self.is_climbable_lut[token_id] = True

            # Foliage / Fragile
            if (
                "leaves" in name
                or "grass" in name
                or "flower" in name
                or "tulip" in name
                or "rose" in name
                or "fern" in name
                or "sapling" in name
                or "bamboo" in name
                or "sugar_cane" in name
                or "lichen" in name
            ):
                self.is_foliage_lut[token_id] = True

            # Resource / Interactive
            if (
                name.endswith("_ore")
                or name.endswith("_ores")
                or "_ore_" in name
                or "chest" in name
                or "furnace" in name
                or "shulker_box" in name
                or "barrel" in name
                or "crafting_table" in name
                or "anvil" in name
                or "spawner" in name
                or "ancient_debris" in name
            ):
                self.is_resource_lut[token_id] = True

            # Collision height
            if self.palette.lut_is_solid[token_id]:
                if "slab" in name or "carpet" in name:
                    self.collision_height_lut[token_id] = 0.5
                else:
                    self.collision_height_lut[token_id] = 1.0

    def adapt(
        self,
        raw_obs: Dict[str, Any],
        player_pos: Tuple[int, int, int],
        target_pos: Tuple[int, int, int],
        cardinal: str = "SOUTH",
        camera_pitch: float = 0.0,
        prev_action: int = -1,
        prev_pos: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Converts raw observation dict and spatial anchors into standard World Model format.

        Returns:
            Dict containing:
              - 'shells': (3, 16, 16, 16, 16) float32 in [0.0, 1.0] (LOD, X, Y, Z, C)
              - 'state': (12,) float32 proprioceptive kinematics
              - 'goal_vec': (3,) float32 normalized egocentric goal direction
              - 'meta': (6,) float32 world and goal metrics
        """
        blocks = raw_obs.get("blocks", raw_obs)
        strides = (1, 2, 4)

        shells = np.zeros((NUM_LODS, GRID_SIZE, GRID_SIZE, GRID_SIZE, NUM_CHANNELS), dtype=np.float32)

        # 1. Process each LOD shell (0, 1, 2)
        for lod_idx, stride in enumerate(strides):
            lod_key = f"lod{lod_idx}"
            b = blocks[lod_key]

            block_ids = b["block_id"]
            passability = b["passability"]
            hazard = b["hazard"]
            hardness = b["hardness"]
            light = b["light_level"]
            visibility = b.get("visibility")

            # Basic channel mapping via fast LUT lookups
            is_solid = self.palette.lut_is_solid[block_ids]
            is_air = (block_ids == 0) | (passability >= 0.95)

            # ch 0: TERRAIN (Solid opaque walkable terrain)
            shells[lod_idx, :, :, :, CH_TERRAIN] = np.where(is_solid, 1.0, 0.0)

            # ch 1: AIR (Passable empty space)
            shells[lod_idx, :, :, :, CH_AIR] = np.where(is_air, 1.0, 0.0)

            # ch 2: HAZARD (Lava, fire, magma)
            shells[lod_idx, :, :, :, CH_HAZARD] = np.clip(hazard, 0.0, 1.0)

            # ch 3: LIQUID (Water and fluid blocks)
            shells[lod_idx, :, :, :, CH_LIQUID] = np.where(self.is_liquid_lut[block_ids], 1.0, 0.0)

            # ch 4: CLIMBABLE (Stairs, ladders, vines)
            shells[lod_idx, :, :, :, CH_CLIMBABLE] = np.where(self.is_climbable_lut[block_ids], 1.0, 0.0)

            # ch 5: FOLIAGE (Grass, flowers, breakables)
            shells[lod_idx, :, :, :, CH_FOLIAGE] = np.where(self.is_foliage_lut[block_ids], 1.0, 0.0)

            # ch 6: WORLD_Y (Normalized world elevation [-64..320] -> [0.0..1.0])
            # Local gy offset: gy in [0..15], player foot is at foot_idx = 6 (for pitch 0)
            foot_index = 6
            gy_indices = np.arange(GRID_SIZE, dtype=np.float32)
            world_y_profile = player_pos[1] + (gy_indices - foot_index) * stride
            norm_y = np.clip((world_y_profile + 64.0) / 384.0, 0.0, 1.0)
            shells[lod_idx, :, :, :, CH_WORLD_Y] = norm_y[None, :, None]

            # ch 7: COLLISION_HEIGHT (0.0 air, 0.5 slab, 1.0 full)
            shells[lod_idx, :, :, :, CH_COLLISION_HEIGHT] = self.collision_height_lut[block_ids]

            # ch 9: RESOURCE (Interactive, chests, ores)
            shells[lod_idx, :, :, :, CH_RESOURCE] = np.where(self.is_resource_lut[block_ids], 1.0, 0.0)

            # ch 10: WALKABLE_SURFACE (Solid floor with >= 2 air blocks above)
            solid_mask = is_solid.astype(bool)
            # Check 2 blocks above in gy dimension
            air_above_1 = np.zeros_like(solid_mask)
            air_above_2 = np.zeros_like(solid_mask)
            air_above_1[:, :-1, :] = is_air[:, 1:, :]
            air_above_2[:, :-2, :] = is_air[:, 2:, :]
            walkable = solid_mask & air_above_1 & air_above_2
            shells[lod_idx, :, :, :, CH_WALKABLE_SURFACE] = np.where(walkable, 1.0, 0.0)

            # ch 11: LIGHT_LEVEL
            shells[lod_idx, :, :, :, CH_LIGHT_LEVEL] = np.clip(light, 0.0, 1.0)

            # ch 12: LETHAL_DROP (Cliff indicator: >= 3 air directly below)
            air_below_1 = np.zeros_like(is_air)
            air_below_2 = np.zeros_like(is_air)
            air_below_3 = np.zeros_like(is_air)
            air_below_1[:, 1:, :] = is_air[:, :-1, :]
            air_below_2[:, 2:, :] = is_air[:, :-2, :]
            air_below_3[:, 3:, :] = is_air[:, :-3, :]
            cliff = is_air & air_below_1 & air_below_2 & air_below_3
            shells[lod_idx, :, :, :, CH_LETHAL_DROP] = np.where(cliff, 1.0, 0.0)

            # ch 13: SURFACE_SHELL (Solid exposed to air, not culled underground)
            if visibility is not None:
                shells[lod_idx, :, :, :, CH_SURFACE_SHELL] = np.where(
                    is_solid & (visibility > 0.0), 1.0, 0.0
                )
            else:
                shells[lod_idx, :, :, :, CH_SURFACE_SHELL] = np.where(is_solid, 1.0, 0.0)

            # ch 14: BLOCK_HARDNESS
            shells[lod_idx, :, :, :, CH_BLOCK_HARDNESS] = np.clip(hardness, 0.0, 1.0)

            # ch 15: OUT_OF_BOUNDS (Void / masked padding)
            shells[lod_idx, :, :, :, CH_OUT_OF_BOUNDS] = np.where(block_ids == 1, 1.0, 0.0)

            # ch 8: WAYPOINT (Inject active goal beacon into voxel shell)
            self._inject_waypoint_channel(
                shell=shells[lod_idx],
                player_pos=player_pos,
                target_pos=target_pos,
                cardinal=cardinal,
                stride=stride,
                foot_index=foot_index
            )

        # 2. Compute Proprioceptive Kinematic State Vector (12,)
        # [vx, vy, vz, dx, dy, dz, sin_yaw, cos_yaw, pitch, health, food, grounded]
        curr_p = np.array(player_pos, dtype=np.float32)
        old_p = np.array(prev_pos if prev_pos is not None else player_pos, dtype=np.float32)
        delta_p = curr_p - old_p

        yaw_deg = CARDINAL_YAW_DEG.get(cardinal, 0.0)
        yaw_rad = math.radians(yaw_deg)

        # Grounded check: inspect foot cell at gx=8, gz=2, gy=5 (1 block below player feet at gy=6)
        grounded = float(shells[0, 8, 5, 2, CH_TERRAIN] > 0.5)

        state = np.array([
            delta_p[0], delta_p[1], delta_p[2],  # vx, vy, vz
            delta_p[0], delta_p[1], delta_p[2],  # dx, dy, dz
            math.sin(yaw_rad), math.cos(yaw_rad),
            float(camera_pitch),
            20.0,  # health
            20.0,  # food
            grounded
        ], dtype=np.float32)

        # 3. Compute Normalized Relative Egocentric Goal Vector (3,)
        dx_world = target_pos[0] - player_pos[0]
        dy_world = target_pos[1] - player_pos[1]
        dz_world = target_pos[2] - player_pos[2]

        vecs = CARDINAL_VECTORS.get(cardinal, CARDINAL_VECTORS["SOUTH"])
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        ego_x = float(dx_world * rx + dz_world * rz)
        ego_y = float(dy_world)
        ego_z = float(dx_world * fx + dz_world * fz)

        raw_goal_vec = np.array([ego_x, ego_y, ego_z], dtype=np.float32)
        goal_dist = float(np.linalg.norm(raw_goal_vec))
        goal_vec_norm = raw_goal_vec / max(goal_dist, 1e-6)

        # 4. Meta Vector (6,) [world_x, world_y, world_z, goal_dx, goal_dy, goal_dz]
        meta = np.array([
            float(player_pos[0]), float(player_pos[1]), float(player_pos[2]),
            ego_x, ego_y, ego_z
        ], dtype=np.float32)

        return {
            "shells": shells,
            "state": state,
            "goal_vec": goal_vec_norm,
            "goal_dist": np.float32(goal_dist),
            "meta": meta,
            "raw_goal_vec": raw_goal_vec
        }

    def _inject_waypoint_channel(
        self,
        shell: np.ndarray,
        player_pos: Tuple[int, int, int],
        target_pos: Tuple[int, int, int],
        cardinal: str,
        stride: int,
        foot_index: int = 6
    ):
        """
        Projects target block into the (16, 16, 16) voxel shell and sets ch 8 to 1.0.
        Pillar spans vertical extent so coarse LODs easily detect distant waypoints.
        """
        dx_world = target_pos[0] - player_pos[0]
        dy_world = target_pos[1] - player_pos[1]
        dz_world = target_pos[2] - player_pos[2]

        vecs = CARDINAL_VECTORS.get(cardinal, CARDINAL_VECTORS["SOUTH"])
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        # Convert to local egocentric offsets
        ego_x = dx_world * rx + dz_world * rz
        ego_y = dy_world
        ego_z = dx_world * fx + dz_world * fz

        # Invert transform: x_local = (gx - 8) * stride -> gx = round(x_local / stride) + 8
        gx = int(round(ego_x / stride)) + 8
        gy = int(round(ego_y / stride)) + foot_index
        gz = int(round(ego_z / stride)) + 2

        if 0 <= gx < GRID_SIZE and 0 <= gz < GRID_SIZE:
            # Pillar beacon: set vertical pillar around target height
            y_min = max(0, gy - 2)
            y_max = min(GRID_SIZE, gy + 3)
            shell[gx, y_min:y_max, gz, CH_WAYPOINT] = 1.0

    @staticmethod
    def to_torch(adapted_obs: Dict[str, Any], device: Optional[Any] = None) -> Dict[str, Any]:
        """Converts adapted NumPy observation dict into PyTorch tensors with batch dimension."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to use to_torch()")

        shells_t = torch.from_numpy(adapted_obs["shells"]).unsqueeze(0)  # (1, 3, 16, 16, 16, 16)
        state_t = torch.from_numpy(adapted_obs["state"]).unsqueeze(0)    # (1, 12)
        goal_t = torch.from_numpy(adapted_obs["goal_vec"]).unsqueeze(0)  # (1, 3)

        if device is not None:
            shells_t = shells_t.to(device)
            state_t = state_t.to(device)
            goal_t = goal_t.to(device)

        return {
            "shells": shells_t,
            "state": state_t,
            "goal_vec": goal_t,
            "goal_dist": float(adapted_obs["goal_dist"])
        }
