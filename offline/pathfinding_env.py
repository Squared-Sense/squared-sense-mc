"""
Gymnasium Offline Minecraft Pathfinding Environment.
Simulates discrete 3D navigation with Bounded BFS target sampling strictly inside B2,
potential-based reward shaping, and complete parity with the live mod and 3D visualizer.
"""

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bfs_target import BoundedBfsTargetGenerator
from .curriculum import DistanceCurriculumScheduler
from .ipc_emitter import MinecraftIpcEmitter
from .lod_extractor import GRID_SIZE, LOD_STRIDES, LodExtractor, WorldVolume
from .palette import BlockPalette, get_palette
from .physics_kernel import MovementPhysicsKernel, MoveOutcome
from .spawn_indexer import SpawnSampler

CARDINAL_ORDER = ["SOUTH", "WEST", "NORTH", "EAST"]

CARDINAL_VECTORS = {
    "SOUTH": {"forward": (0, 1),  "right": (-1, 0)},
    "WEST":  {"forward": (-1, 0), "right": (0, -1)},
    "NORTH": {"forward": (0, -1), "right": (1, 0)},
    "EAST":  {"forward": (1, 0),  "right": (0, 1)}
}

# Discrete Actions
ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_TURN_LEFT = 4
ACTION_TURN_RIGHT = 5

ACTION_NAMES = [
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT"
]


class MinecraftPathfindingEnv(gym.Env):
    """
    Gymnasium environment for offline voxel pathfinding pretraining.
    """

    metadata = {"render_modes": ["human", "ipc", "rgb_array"]}

    def __init__(
        self,
        world_volume: WorldVolume,
        spawn_sampler: SpawnSampler,
        lod_extractor: Optional[LodExtractor] = None,
        palette_obj: Optional[BlockPalette] = None,
        allow_turns: bool = True,
        curriculum_scheduler: Optional[DistanceCurriculumScheduler] = None,
        render_mode: Optional[str] = None,
        progress_reward_alpha: float = 1.0,
        step_penalty: float = -0.02,
        blocked_penalty: float = -0.05,
        goal_bonus: float = 15.0,
        fatal_penalty: float = -10.0,
        goal_reach_radius: float = 1.2,
        mask_fatal: bool = False
    ):
        super().__init__()
        self.world = world_volume
        self.spawn_sampler = spawn_sampler
        self.palette = palette_obj or self.world.palette
        self.extractor = lod_extractor or LodExtractor(self.world, self.palette)
        self.kernel = MovementPhysicsKernel(self.world, self.palette)
        self.target_gen = BoundedBfsTargetGenerator(self.kernel)
        self.curriculum = curriculum_scheduler or DistanceCurriculumScheduler()

        self.allow_turns = allow_turns
        self.mask_fatal = mask_fatal
        self.num_actions = 6 if allow_turns else 4
        self.action_space = spaces.Discrete(self.num_actions)

        self.render_mode = render_mode
        self.emitter: Optional[MinecraftIpcEmitter] = None
        if self.render_mode in ("human", "ipc"):
            self.emitter = MinecraftIpcEmitter()

        # Reward hyperparameters
        self.alpha = progress_reward_alpha
        self.step_penalty = step_penalty
        self.blocked_penalty = blocked_penalty
        self.goal_bonus = goal_bonus
        self.fatal_penalty = fatal_penalty
        self.goal_reach_radius = goal_reach_radius

        # Define Gymnasium Observation Space
        grid_shape = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
        block_spaces = {}
        entity_spaces = {}

        for lod in range(3):
            lod_key = f"lod{lod}"
            block_spaces[lod_key] = spaces.Dict({
                "block_id": spaces.Box(low=0, high=self.palette.vocab_size - 1, shape=grid_shape, dtype=np.uint16),
                "occupancy": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "passability": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "hazard": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "hardness": spaces.Box(low=0.0, high=100.0, shape=grid_shape, dtype=np.float32),
                "light_level": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "fluid_flow_x": spaces.Box(low=-1.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "fluid_flow_z": spaces.Box(low=-1.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "visibility": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32)
            })
            entity_spaces[lod_key] = spaces.Dict({
                "entity_count": spaces.Box(low=0, high=255, shape=grid_shape, dtype=np.uint8),
                "entity_type_id": spaces.Box(low=0, high=255, shape=grid_shape, dtype=np.uint8),
                "is_hostile": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "total_health": spaces.Box(low=0.0, high=500.0, shape=grid_shape, dtype=np.float32)
            })

        self.observation_space = spaces.Dict({
            "blocks": spaces.Dict(block_spaces),
            "entities": spaces.Dict(entity_spaces),
            "goal_delta_xyz": spaces.Box(low=-128.0, high=128.0, shape=(3,), dtype=np.float32),
            "goal_dist": spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            "prev_action": spaces.Box(low=-1, high=self.num_actions - 1, shape=(1,), dtype=np.int64),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.bool_)
        })

        # Episode state variables
        self.player_pos: Tuple[int, int, int] = (0, 0, 0)
        self.spawn_pos: Tuple[int, int, int] = (0, 0, 0)
        self.target_pos: Tuple[int, int, int] = (0, 0, 0)
        self.cardinal: str = "SOUTH"
        self.camera_pitch: float = 0.0
        self.l_bfs: int = 0
        self.optimal_path: List[Tuple[int, int, int]] = []
        self.visited_trail: List[Tuple[int, int, int]] = []

        self.step_count: int = 0
        self.max_steps: int = 50
        self.prev_dist: float = 0.0
        self.prev_action: int = -1
        self.episode_return: float = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # 1. Sample spawn point from curriculum 60/25/15 index
        self.spawn_pos = self.spawn_sampler.sample(rng=rng)
        self.player_pos = self.spawn_pos
        self.visited_trail = [self.player_pos]

        # Randomize initial heading or default to SOUTH
        self.cardinal = rng.choice(CARDINAL_ORDER)
        self.camera_pitch = 0.0

        # 2. Bounded BFS Target Generation strictly inside B2
        target_dist_range = self.curriculum.get_distance_range()
        foot_index = 6
        self.target_pos, self.l_bfs, self.optimal_path = self.target_gen.generate_target(
            spawn=self.spawn_pos,
            cardinal=self.cardinal,
            foot_index=foot_index,
            target_dist_range=target_dist_range,
            rng=rng
        )

        # 3. Setup episode limits and metrics
        self.step_count = 0
        self.max_steps = max(15, 3 * self.l_bfs)
        self.prev_dist = self._compute_euclidean_distance(self.player_pos, self.target_pos)
        self.prev_action = -1
        self.episode_return = 0.0

        # 4. Build observation and info
        obs = self._get_observation()
        info = self._get_info(outcome=MoveOutcome.SUCCESS)

        if self.emitter:
            self.render()

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = self.step_penalty  # Default step penalty (-0.02)
        terminated = False
        truncated = False
        outcome = MoveOutcome.SUCCESS

        vecs = CARDINAL_VECTORS[self.cardinal]
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        # Action Execution
        if not self.allow_turns and action >= 4:
            raise ValueError(f"Invalid action {action} when allow_turns is False")

        if action == ACTION_FORWARD:
            nx, ny, nz, outcome, _ = self.kernel.step(
                self.player_pos[0], self.player_pos[1], self.player_pos[2], fx, fz
            )
            self.player_pos = (nx, ny, nz)
        elif action == ACTION_BACKWARD:
            nx, ny, nz, outcome, _ = self.kernel.step(
                self.player_pos[0], self.player_pos[1], self.player_pos[2], -fx, -fz
            )
            self.player_pos = (nx, ny, nz)
        elif action == ACTION_LEFT:
            nx, ny, nz, outcome, _ = self.kernel.step(
                self.player_pos[0], self.player_pos[1], self.player_pos[2], -rx, -rz
            )
            self.player_pos = (nx, ny, nz)
        elif action == ACTION_RIGHT:
            nx, ny, nz, outcome, _ = self.kernel.step(
                self.player_pos[0], self.player_pos[1], self.player_pos[2], rx, rz
            )
            self.player_pos = (nx, ny, nz)
        elif action == ACTION_TURN_LEFT:
            curr_idx = CARDINAL_ORDER.index(self.cardinal)
            self.cardinal = CARDINAL_ORDER[(curr_idx - 1) % 4]
            outcome = MoveOutcome.SUCCESS
        elif action == ACTION_TURN_RIGHT:
            curr_idx = CARDINAL_ORDER.index(self.cardinal)
            self.cardinal = CARDINAL_ORDER[(curr_idx + 1) % 4]
            outcome = MoveOutcome.SUCCESS
        else:
            raise ValueError(f"Invalid action {action}")

        self.visited_trail.append(self.player_pos)
        self.prev_action = action

        # Outcome evaluation & Reward Shaping
        curr_dist = self._compute_euclidean_distance(self.player_pos, self.target_pos)

        if outcome == MoveOutcome.BLOCKED:
            reward += self.blocked_penalty
        elif outcome in (MoveOutcome.FATAL_FALL, MoveOutcome.FATAL_HAZARD, MoveOutcome.FATAL_VOID):
            reward += self.fatal_penalty
            terminated = True
            self.curriculum.record_episode(success=False)
        else:
            # Potential-based progress reward: +alpha * (d_{t-1} - d_t)
            progress = self.alpha * (self.prev_dist - curr_dist)
            reward += progress

        self.prev_dist = curr_dist

        # Check Goal Reached
        if not terminated and curr_dist <= self.goal_reach_radius:
            reward += self.goal_bonus
            terminated = True
            self.curriculum.record_episode(success=True)

        # Check Timeout Truncation
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.curriculum.record_episode(success=False)

        self.episode_return += reward
        obs = self._get_observation()
        info = self._get_info(outcome=outcome)

        if self.emitter:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, Any]:
        """Extracts observation tensors and egocentric relative goal vector."""
        raw_obs = self.extractor.extract_observation(
            player_pos=self.player_pos,
            cardinal=self.cardinal,
            camera_pitch=self.camera_pitch,
            apply_visibility_culling=True
        )

        # Relative Goal Delta in Egocentric Frame
        dx = self.target_pos[0] - self.player_pos[0]
        dy = self.target_pos[1] - self.player_pos[1]
        dz = self.target_pos[2] - self.player_pos[2]

        vecs = CARDINAL_VECTORS[self.cardinal]
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        local_x = float(dx * rx + dz * rz)
        local_y = float(dy)
        local_z = float(dx * fx + dz * fz)

        goal_delta_xyz = np.array([local_x, local_y, local_z], dtype=np.float32)
        goal_dist = np.array([self.prev_dist], dtype=np.float32)
        prev_act = np.array([self.prev_action], dtype=np.int64)
        action_mask = self.get_action_mask()

        return {
            "blocks": raw_obs["blocks"],
            "entities": raw_obs["entities"],
            "goal_delta_xyz": goal_delta_xyz,
            "goal_dist": goal_dist,
            "prev_action": prev_act,
            "action_mask": action_mask
        }

    def _get_info(self, outcome: MoveOutcome) -> Dict[str, Any]:
        is_success = (self.prev_dist <= self.goal_reach_radius)
        efficiency = (self.l_bfs / max(1, self.step_count)) if self.step_count > 0 else 1.0

        return {
            "is_success": is_success,
            "outcome": outcome.name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "l_bfs": self.l_bfs,
            "goal_dist": round(self.prev_dist, 2),
            "episode_return": round(self.episode_return, 3),
            "efficiency_ratio": round(efficiency, 3),
            "curriculum_stage": self.curriculum.stage,
            "rolling_success_rate": round(self.curriculum.get_success_rate(), 2),
            "player_pos": self.player_pos,
            "target_pos": self.target_pos,
            "cardinal": self.cardinal,
            "optimal_path": self.optimal_path,
            "visited_trail": self.visited_trail
        }

    def get_action_mask(self) -> np.ndarray:
        """
        Computes action mask (True for legal unblocked non-fatal moves).
        """
        mask = np.ones(self.num_actions, dtype=np.bool_)
        vecs = CARDINAL_VECTORS[self.cardinal]
        fx, fz = vecs["forward"]
        rx, rz = vecs["right"]

        candidate_deltas = [
            (fx, fz),    # FORWARD
            (-fx, -fz),  # BACKWARD
            (-rx, -rz),  # LEFT
            (rx, rz)     # RIGHT
        ]

        px, py, pz = self.player_pos
        for act in range(4):
            dx, dz = candidate_deltas[act]
            _, _, _, outcome, _ = self.kernel.step(px, py, pz, dx, dz)
            if outcome == MoveOutcome.BLOCKED:
                mask[act] = False
            elif self.mask_fatal and outcome in (MoveOutcome.FATAL_FALL, MoveOutcome.FATAL_HAZARD, MoveOutcome.FATAL_VOID):
                mask[act] = False

        # Turn actions are always legal if enabled
        if self.allow_turns:
            mask[ACTION_TURN_LEFT] = True
            mask[ACTION_TURN_RIGHT] = True

        return mask

    def _compute_euclidean_distance(self, p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> float:
        return float(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))

    def render(self):
        """Streams live observation into MMF for real-time 3D visualizer display."""
        if not self.emitter:
            return

        obs = self.extractor.extract_observation(
            player_pos=self.player_pos,
            cardinal=self.cardinal,
            camera_pitch=self.camera_pitch
        )
        self.emitter.emit_observation(obs, game_tick=self.step_count * 2)

    def close(self):
        if self.emitter:
            self.emitter.close()
            self.emitter = None

