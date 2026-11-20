"""
Squared Sense MC - Offline RL & Data Pipeline Package
"""

from .palette import BlockPalette, get_palette
from .mca_baker import bake_region, bake_world
from .spawn_indexer import SpawnSampler, index_region_spawns
from .lod_extractor import WorldVolume, LodExtractor
from .physics_kernel import MovementPhysicsKernel, MoveOutcome
from .bfs_target import BoundedBfsTargetGenerator
from .curriculum import DistanceCurriculumScheduler
from .pathfinding_env import MinecraftPathfindingEnv

__all__ = [
    "BlockPalette",
    "get_palette",
    "bake_region",
    "bake_world",
    "SpawnSampler",
    "index_region_spawns",
    "WorldVolume",
    "LodExtractor",
    "MovementPhysicsKernel",
    "MoveOutcome",
    "BoundedBfsTargetGenerator",
    "DistanceCurriculumScheduler",
    "MinecraftPathfindingEnv",
]

