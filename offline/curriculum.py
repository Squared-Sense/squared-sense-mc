"""
Distance Curriculum Scheduler for Minecraft 3D Pathfinding.
Gradually expands target sampling distance as agent pathfinding proficiency improves:
- Stage 0 (B0 horizon): Distance 2 .. 6 blocks
- Stage 1 (B1 horizon): Distance 6 .. 12 blocks
- Stage 2 (Mid B2):     Distance 12 .. 24 blocks
- Stage 3 (Full B2):    Distance 24 .. 45 blocks
"""

from collections import deque
from typing import Dict, Optional, Tuple


class DistanceCurriculumScheduler:
    """
    Manages curriculum distance stages and adaptive advancement based on rolling success rate.
    """

    STAGES: Dict[int, Tuple[int, int]] = {
        0: (2, 6),     # Stage 0: Inside B0
        1: (6, 12),    # Stage 1: Inside B1
        2: (12, 24),   # Stage 2: Mid B2
        3: (24, 45)    # Stage 3: Full B2 horizon
    }

    def __init__(
        self,
        initial_stage: int = 0,
        window_size: int = 50,
        advance_threshold: float = 0.80,
        demote_threshold: float = 0.25,
        min_episodes_before_advance: int = 25
    ):
        self.stage = initial_stage
        self.window_size = window_size
        self.advance_threshold = advance_threshold
        self.demote_threshold = demote_threshold
        self.min_episodes = min_episodes_before_advance

        self.history: deque = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_successes = 0

    def get_distance_range(self) -> Tuple[int, int]:
        """Returns (min_dist, max_dist) for current curriculum stage."""
        return self.STAGES.get(self.stage, (2, 6))

    def record_episode(self, success: bool) -> bool:
        """
        Records an episode outcome (True for goal reached, False for fatal/timeout).
        Returns True if a curriculum stage transition occurred.
        """
        self.total_episodes += 1
        if success:
            self.total_successes += 1
        self.history.append(1 if success else 0)

        transition = False
        if len(self.history) >= self.min_episodes:
            success_rate = sum(self.history) / len(self.history)

            # Check for stage advancement
            if success_rate >= self.advance_threshold and self.stage < max(self.STAGES.keys()):
                self.stage += 1
                self.history.clear()
                transition = True
                print(
                    f"[CurriculumScheduler] Advanced to Stage {self.stage} "
                    f"(Target distance: {self.get_distance_range()}) with success rate {success_rate * 100:.1f}%"
                )

            # Check for stage demotion
            elif success_rate <= self.demote_threshold and self.stage > 0:
                self.stage -= 1
                self.history.clear()
                transition = True
                print(
                    f"[CurriculumScheduler] Demoted to Stage {self.stage} "
                    f"(Target distance: {self.get_distance_range()}) with success rate {success_rate * 100:.1f}%"
                )

        return transition

    def get_success_rate(self) -> float:
        """Returns recent rolling success rate [0.0, 1.0]."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def set_stage(self, stage: int):
        """Manually sets the curriculum stage [0 .. 3]."""
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage {stage}. Available: {list(self.STAGES.keys())}")
        self.stage = stage
        self.history.clear()

