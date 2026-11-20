"""
Bounded BFS Target Generator (Strictly Inside B2).
Generates guaranteed reachable, visible target blocks by running 3D BFS
clamped within the spatial bounding box of LOD 2 (B2).
Provides optimal path length L_bfs and reference trajectory.
"""

from collections import defaultdict, deque
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .physics_kernel import MovementPhysicsKernel, MoveOutcome


def compute_b2_bounding_box(
    spawn: Tuple[int, int, int],
    cardinal: str = "SOUTH",
    foot_index: int = 6
) -> Tuple[int, int, int, int, int, int]:
    """
    Computes (min_wx, max_wx, min_wy, max_wy, min_wz, max_wz) for LOD 2 (stride 4, 16x16x16).
    """
    x0, y0, z0 = spawn
    stride = 4

    min_lx = (0 - 8) * stride           # -32
    max_lx = (15 - 8) * stride + (stride - 1)  # +31
    min_lz = (0 - 2) * stride           # -8
    max_lz = (15 - 2) * stride + (stride - 1)  # +55
    min_ly = (0 - foot_index) * stride  # -24 (for foot_index 6)
    max_ly = (15 - foot_index) * stride + (stride - 1)  # +39

    min_wy = y0 + min_ly
    max_wy = y0 + max_ly

    if cardinal == "SOUTH":
        min_wx = x0 - max_lx
        max_wx = x0 - min_lx
        min_wz = z0 + min_lz
        max_wz = z0 + max_lz
    elif cardinal == "NORTH":
        min_wx = x0 + min_lx
        max_wx = x0 + max_lx
        min_wz = z0 - max_lz
        max_wz = z0 - min_lz
    elif cardinal == "WEST":
        min_wx = x0 - max_lz
        max_wx = x0 - min_lz
        min_wz = z0 - max_lx
        max_wz = z0 - min_lx
    else:  # EAST
        min_wx = x0 + min_lz
        max_wx = x0 + max_lz
        min_wz = z0 + min_lx
        max_wz = z0 + max_lx

    return min_wx, max_wx, min_wy, max_wy, min_wz, max_wz


class BoundedBfsTargetGenerator:
    """
    Generates verified traversable goals within the agent's B2 observation field.
    """

    def __init__(self, physics_kernel: MovementPhysicsKernel, max_explored_nodes: int = 5000):
        self.kernel = physics_kernel
        self.max_explored_nodes = max_explored_nodes

    def generate_target(
        self,
        spawn: Tuple[int, int, int],
        cardinal: str = "SOUTH",
        foot_index: int = 6,
        target_dist_range: Tuple[int, int] = (6, 16),
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[Tuple[int, int, int], int, List[Tuple[int, int, int]]]:
        """
        Runs bounded 3D BFS from spawn strictly inside B2.
        Returns: (target_position, L_bfs, optimal_path).
        """
        if rng is None:
            rng = np.random.default_rng()

        min_d, max_d = target_dist_range
        b2_box = compute_b2_bounding_box(spawn, cardinal, foot_index)
        min_wx, max_wx, min_wy, max_wy, min_wz, max_wz = b2_box

        queue = deque([spawn])
        dist_map: Dict[Tuple[int, int, int], int] = {spawn: 0}
        parent_map: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {spawn: None}
        reachable_by_dist: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        reachable_by_dist[0].append(spawn)

        # Explore 4 horizontal moves
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while queue and len(dist_map) < self.max_explored_nodes:
            curr = queue.popleft()
            d = dist_map[curr]

            if d >= max_d + 4:
                # Reached sufficient depth beyond target horizon
                continue

            cx, cy, cz = curr
            for dx, dz in moves:
                nx, ny, nz, outcome, _ = self.kernel.step(cx, cy, cz, dx, dz)

                if outcome in (MoveOutcome.SUCCESS, MoveOutcome.STEP_UP, MoveOutcome.FALL):
                    # Clamp strictly inside B2 spatial bounding volume
                    if (
                        min_wx <= nx <= max_wx
                        and min_wy <= ny <= max_wy
                        and min_wz <= nz <= max_wz
                    ):
                        next_pos = (nx, ny, nz)
                        if next_pos not in dist_map:
                            new_dist = d + 1
                            dist_map[next_pos] = new_dist
                            parent_map[next_pos] = curr
                            reachable_by_dist[new_dist].append(next_pos)
                            queue.append(next_pos)

        # Candidate selection matching curriculum distance range [min_d, max_d]
        candidates: List[Tuple[int, int, int]] = []
        for d_target in range(min_d, max_d + 1):
            if d_target in reachable_by_dist:
                candidates.extend(reachable_by_dist[d_target])

        if not candidates:
            # Fallback to the farthest reachable distance > 0
            available_dists = [d for d in sorted(reachable_by_dist.keys()) if d > 0]
            if available_dists:
                best_d = available_dists[-1]
                candidates = reachable_by_dist[best_d]
            else:
                # Agent completely enclosed (0 moves possible)
                return spawn, 0, [spawn]

        # Pick random candidate from chosen distance tier
        pick_idx = rng.integers(0, len(candidates))
        target_pos = candidates[pick_idx]
        l_bfs = dist_map[target_pos]

        # Reconstruct optimal path from spawn to target
        path = []
        curr_trace = target_pos
        while curr_trace is not None:
            path.append(curr_trace)
            curr_trace = parent_map[curr_trace]
        path.reverse()

        return target_pos, l_bfs, path

