"""
Inference Telemetry Specification & Binary Serializer for 3D Visualizer Inspection.

Extends the Shared Memory (MMF) Header Page (Page 0, offset 256) with rich RL telemetry:
  - 3D Target block coordinates (world + relative)
  - Agent breadcrumb trail and optimal BFS trajectory
  - Real-time Critic value estimate V(s), step reward, and action probabilities
  - Checkpoint metadata (policy/value losses, training steps, success rate)
"""

import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

TELEMETRY_MAGIC = 0x54454C4D  # "TELM"
TELEMETRY_VERSION = 1
OFF_TELEMETRY_BASE = 256
MAX_PATH_POINTS = 64

# Format strings:
# Header: Magic(I), Version(I), Enabled(I), Step(I), TgtX(d), TgtY(d), TgtZ(d),
#         TgtDx(f), TgtDy(f), TgtDz(f), Dist(f), Reward(f), Return(f), Critic(f),
#         NumAct(I), ActProbs(12f), ChosenAct(i), L_BFS(I), TrailCnt(I), BfsCnt(I),
#         CkptSteps(I), CkptPLoss(f), CkptVLoss(f), CkptSuccess(f)
HEADER_FMT = "<IIII ddd fff fff f I 12f i I I I I fff"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class InferenceTelemetry:
    """Helper for packing and unpacking inference telemetry directly into MMF Page 0."""

    @staticmethod
    def pack(
        target_pos: Tuple[int, int, int] = (0, 0, 0),
        player_pos: Tuple[int, int, int] = (0, 0, 0),
        target_rel: Optional[Tuple[float, float, float]] = None,
        world_delta: Optional[Tuple[float, float, float]] = None,
        remaining_dist: Optional[float] = None,
        target_dist: Optional[float] = None,
        step_reward: float = 0.0,
        episode_return: float = 0.0,
        step_count: int = 0,
        critic_value: float = 0.0,
        action_probs: Optional[List[float]] = None,
        action_probabilities: Optional[List[float]] = None,
        chosen_action: int = 0,
        l_bfs: int = 0,
        visited_trail: Optional[List[Tuple[int, int, int]]] = None,
        breadcrumbs: Optional[List[Tuple[int, int, int]]] = None,
        bfs_path: Optional[List[Tuple[int, int, int]]] = None,
        optimal_path: Optional[List[Tuple[int, int, int]]] = None,
        checkpoint_steps: int = 0,
        checkpoint_policy_loss: float = 0.0,
        checkpoint_value_loss: float = 0.0,
        checkpoint_success_rate: float = 0.0,
        checkpoint_loss: Optional[float] = None
    ) -> bytes:
        """Packs inference telemetry into byte buffer for MMF Page 0."""
        # Resolve aliases
        rel = world_delta if world_delta is not None else (target_rel if target_rel is not None else (0.0, 0.0, 0.0))
        dist = target_dist if target_dist is not None else (remaining_dist if remaining_dist is not None else 0.0)
        probs = action_probabilities if action_probabilities is not None else (action_probs if action_probs is not None else [0.0] * 12)
        trail_pts = breadcrumbs if breadcrumbs is not None else (visited_trail if visited_trail is not None else [])
        path_pts = optimal_path if optimal_path is not None else (bfs_path if bfs_path is not None else [])
        p_loss = checkpoint_loss if checkpoint_loss is not None else checkpoint_policy_loss

        # Pad action probabilities to 12 floats
        probs_12 = list(probs[:12]) + [0.0] * max(0, 12 - len(probs))

        # Sample or truncate trails to MAX_PATH_POINTS
        trail = trail_pts[-MAX_PATH_POINTS:]
        bfs = path_pts[:MAX_PATH_POINTS]

        trail_cnt = len(trail)
        bfs_cnt = len(bfs)

        header_bytes = struct.pack(
            HEADER_FMT,
            TELEMETRY_MAGIC,
            TELEMETRY_VERSION,
            1,  # Enabled
            int(step_count),
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
            float(rel[0]), float(rel[1]), float(rel[2]),
            float(dist),
            float(step_reward),
            float(episode_return),
            float(critic_value),
            int(len(probs)),
            *probs_12,
            int(chosen_action),
            int(l_bfs),
            int(trail_cnt),
            int(bfs_cnt),
            int(checkpoint_steps),
            float(p_loss),
            float(checkpoint_value_loss),
            float(checkpoint_success_rate)
        )

        # Pack relative path deltas from player position as int16
        # To avoid overflow, clamp to int16 range [-32768..32767]
        trail_coords = bytearray(MAX_PATH_POINTS * 3 * 2)
        for i, (wx, wy, wz) in enumerate(trail):
            dx = int(round(max(-32000, min(32000, wx - player_pos[0]))))
            dy = int(round(max(-32000, min(32000, wy - player_pos[1]))))
            dz = int(round(max(-32000, min(32000, wz - player_pos[2]))))
            struct.pack_into("<hhh", trail_coords, i * 6, dx, dy, dz)

        bfs_coords = bytearray(MAX_PATH_POINTS * 3 * 2)
        for i, (wx, wy, wz) in enumerate(bfs):
            dx = int(round(max(-32000, min(32000, wx - player_pos[0]))))
            dy = int(round(max(-32000, min(32000, wy - player_pos[1]))))
            dz = int(round(max(-32000, min(32000, wz - player_pos[2]))))
            struct.pack_into("<hhh", bfs_coords, i * 6, dx, dy, dz)

        return header_bytes + bytes(trail_coords) + bytes(bfs_coords)

    @staticmethod
    def unpack(data: bytes) -> Optional[Dict[str, Any]]:
        """Unpacks inference telemetry from byte buffer."""
        if len(data) < HEADER_SIZE:
            return None

        unpacked = struct.unpack_from(HEADER_FMT, data, 0)
        magic, ver, enabled = unpacked[0], unpacked[1], unpacked[2]

        if magic != TELEMETRY_MAGIC or enabled == 0:
            return None

        step = unpacked[3]
        tgt_x, tgt_y, tgt_z = unpacked[4], unpacked[5], unpacked[6]
        tgt_dx, tgt_dy, tgt_dz = unpacked[7], unpacked[8], unpacked[9]
        dist = unpacked[10]
        reward = unpacked[11]
        ep_return = unpacked[12]
        critic = unpacked[13]
        num_act = unpacked[14]
        probs = list(unpacked[15:27])[:num_act]
        chosen = unpacked[27]
        l_bfs = unpacked[28]
        trail_cnt = unpacked[29]
        bfs_cnt = unpacked[30]
        ckpt_steps = unpacked[31]
        ckpt_ploss = unpacked[32]
        ckpt_vloss = unpacked[33]
        ckpt_success = unpacked[34]

        # Unpack trail coords
        offset = HEADER_SIZE
        trail_points = []
        for i in range(min(trail_cnt, MAX_PATH_POINTS)):
            dx, dy, dz = struct.unpack_from("<hhh", data, offset + i * 6)
            trail_points.append([dx, dy, dz])

        offset += MAX_PATH_POINTS * 6
        bfs_points = []
        for i in range(min(bfs_cnt, MAX_PATH_POINTS)):
            dx, dy, dz = struct.unpack_from("<hhh", data, offset + i * 6)
            bfs_points.append([dx, dy, dz])

        return {
            "step": step,
            "targetPos": [tgt_x, tgt_y, tgt_z],
            "targetRel": [round(tgt_dx, 2), round(tgt_dy, 2), round(tgt_dz, 2)],
            "remainingDist": round(dist, 2),
            "stepReward": round(reward, 3),
            "episodeReturn": round(ep_return, 3),
            "criticValue": round(critic, 3),
            "numActions": num_act,
            "actionProbs": [round(p, 4) for p in probs],
            "chosenAction": chosen,
            "lBfs": l_bfs,
            "trailPoints": trail_points,
            "bfsPoints": bfs_points,
            "checkpoint": {
                "steps": ckpt_steps,
                "policyLoss": round(ckpt_ploss, 4),
                "valueLoss": round(ckpt_vloss, 4),
                "successRate": round(ckpt_success, 3)
            }
        }

