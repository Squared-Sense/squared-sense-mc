"""
Deterministic Movement Physics Kernel for Minecraft 3D Pathfinding.
Enforces 1x2x1 player hitbox, 1-block step-up, gravity falling, and fatal hazard/fall detection.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np

from .mca_baker import WORLD_MIN_Y
from .palette import BlockPalette, get_palette


class MoveOutcome(Enum):
    SUCCESS = 0         # Standard flat move (same Y)
    STEP_UP = 1         # Stepped up 1 block onto solid obstacle
    FALL = 2            # Stepped off ledge and landed safely (fall distance <= 5)
    BLOCKED = 3         # Move blocked by wall, low ceiling, or lack of step-up clearance
    FATAL_FALL = 4      # Fell > 5 blocks
    FATAL_HAZARD = 5    # Landed on or walked into lava/fire/hazard
    FATAL_VOID = 6      # Fell into void below world minimum Y


class MovementPhysicsKernel:
    """
    Simulates discrete 1x2x1 player locomotion in Minecraft voxel worlds.
    """

    def __init__(self, world_volume, palette_obj: Optional[BlockPalette] = None, max_safe_fall: int = 5):
        self.world = world_volume
        self.palette = palette_obj or get_palette()
        self.max_safe_fall = max_safe_fall

    def is_passable_air(self, block_id: int) -> bool:
        """Returns True if block can be occupied by player body (air, grass, flowers)."""
        return self.palette.lut_passability[block_id] >= 0.5 and self.palette.lut_hazard[block_id] == 0.0

    def is_solid_floor(self, block_id: int) -> bool:
        """Returns True if block can support player feet without hazard."""
        return self.palette.lut_is_solid[block_id] and self.palette.lut_hazard[block_id] == 0.0

    def is_hazard(self, block_id: int) -> bool:
        """Returns True if block is dangerous (lava, fire, magma, cactus)."""
        return self.palette.lut_hazard[block_id] > 0.0

    def is_valid_standing_position(self, x: int, y: int, z: int) -> bool:
        """
        Validates 1x2x1 player hitbox clearance at feet position (x, y, z):
        - Floor at y-1 must be solid and non-hazard.
        - Feet at y must be passable and non-hazard.
        - Head at y+1 must be passable and non-hazard.
        """
        floor_b = self.world.get_block(x, y - 1, z)
        if not self.is_solid_floor(floor_b):
            return False

        feet_b = self.world.get_block(x, y, z)
        if not self.is_passable_air(feet_b):
            return False

        head_b = self.world.get_block(x, y + 1, z)
        if not self.is_passable_air(head_b):
            return False

        return True

    def step(
        self,
        x: int,
        y: int,
        z: int,
        dx: int,
        dz: int
    ) -> Tuple[int, int, int, MoveOutcome, int]:
        """
        Simulates moving 1 block horizontally from (x, y, z) along (dx, dz).
        Returns: (next_x, next_y, next_z, outcome, fall_distance).
        """
        nx, nz = x + dx, z + dz

        # Query relevant blocks
        target_feet_b = self.world.get_block(nx, y, nz)

        # -------------------------------------------------------------
        # 1. Step-UP check (if obstacle in front is solid)
        # -------------------------------------------------------------
        if self.palette.lut_is_solid[target_feet_b]:
            # Can step up 1 block if:
            # - Current position has headroom to jump: (x, y+2, z) is passable
            # - Target position has headroom for feet: (nx, y+1, nz) is passable
            # - Target position has headroom for head: (nx, y+2, nz) is passable
            # - Target floor (nx, y, nz) is non-hazard
            current_headroom = self.world.get_block(x, y + 2, z)
            target_step_feet = self.world.get_block(nx, y + 1, nz)
            target_step_head = self.world.get_block(nx, y + 2, nz)

            if (
                self.is_passable_air(current_headroom)
                and self.is_passable_air(target_step_feet)
                and self.is_passable_air(target_step_head)
                and not self.is_hazard(target_feet_b)
            ):
                return nx, y + 1, nz, MoveOutcome.STEP_UP, 0
            else:
                return x, y, z, MoveOutcome.BLOCKED, 0

        # -------------------------------------------------------------
        # 2. Hazard check at entry level
        # -------------------------------------------------------------
        if self.is_hazard(target_feet_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        # Check head clearance at entry level
        target_head_b = self.world.get_block(nx, y + 1, nz)
        if self.is_hazard(target_head_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        if not self.is_passable_air(target_head_b):
            # Blocked by low ceiling
            return x, y, z, MoveOutcome.BLOCKED, 0

        if not self.is_passable_air(target_feet_b):
            # Target feet block is not passable (e.g. non-solid impeding wall)
            return x, y, z, MoveOutcome.BLOCKED, 0

        # -------------------------------------------------------------
        # 3. Flat Move Check (Floor at y-1)
        # -------------------------------------------------------------
        target_floor_b = self.world.get_block(nx, y - 1, nz)
        if self.is_hazard(target_floor_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        if self.palette.lut_is_solid[target_floor_b]:
            # Solid floor -> successful flat move
            return nx, y, nz, MoveOutcome.SUCCESS, 0

        # -------------------------------------------------------------
        # 4. Gravity & Fall Resolution
        # -------------------------------------------------------------
        fall_dist = 0
        curr_y = y

        while True:
            fall_dist += 1
            drop_feet_y = curr_y - 1
            drop_floor_y = drop_feet_y - 1

            if drop_feet_y < WORLD_MIN_Y:
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_VOID, fall_dist

            drop_feet_b = self.world.get_block(nx, drop_feet_y, nz)
            if self.is_hazard(drop_feet_b):
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_HAZARD, fall_dist

            drop_floor_b = self.world.get_block(nx, drop_floor_y, nz)
            if self.is_hazard(drop_floor_b):
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_HAZARD, fall_dist

            if self.palette.lut_is_solid[drop_floor_b]:
                # Landed on solid ground
                if fall_dist > self.max_safe_fall:
                    return nx, drop_feet_y, nz, MoveOutcome.FATAL_FALL, fall_dist
                else:
                    return nx, drop_feet_y, nz, MoveOutcome.FALL, fall_dist

            # Still falling through air
            curr_y = drop_feet_y
            if fall_dist > 15:
                # Terminal void / fatal fall
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_FALL, fall_dist

