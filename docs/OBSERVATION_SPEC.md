# Observation Specification & Commands

This document specifies the data formats, command interfaces, and integration patterns for consuming Squared Sense MC observations in Python / PyTorch RL pipelines.

---

## 1. LOD (Level of Detail) System

Observations use a fixed $16\times 16\times 16$ tensor volume across 3 hierarchical LOD levels:

| Level | Tensor Shape | Stride (Scale) | Physical World Coverage | Sub-Volume per Cell |
|---|---|---|---|---|
| **$B_0$ (LOD 0)** | $16\times 16\times 16$ | $1$ block | $16\times 16\times 16$ meters | $1\times 1\times 1$ ($1$ block) |
| **$B_1$ (LOD 1)** | $16\times 16\times 16$ | $2$ blocks | $32\times 32\times 32$ meters | $2\times 2\times 2$ ($8$ blocks) |
| **$B_2$ (LOD 2)** | $16\times 16\times 16$ | $4$ blocks | $64\times 64\times 64$ meters | $4\times 4\times 4$ ($64$ blocks) |

### Priority Downsampling Algorithm (Task 7)
For downsampled levels ($B_1$ and $B_2$), naive majority voting is replaced with a **priority-tier hierarchy**. If a $2\times 2\times 2$ (LOD 1) or $4\times 4\times 4$ (LOD 2) volume contains critical blocks (e.g., a single lava block or chest), they are preserved in the categorical block ID channel even when in the minority:

```
HAZARD (Rank 6) > INTERACTIVE / CRITICAL (Rank 5) > LIQUID (Rank 4) > FOLIAGE (Rank 3) > SOLID (Rank 2) > AIR (Rank 1)
```

#### Priority Tiers

| Tier | Rank | Included Block Types | Example Behavior |
|---|---|---|---|
| **HAZARD** | 6 | Lava, Fire, Soul Fire, Campfires, Magma Block, Cactus, Sweet Berry Bush, Wither Rose, Powder Snow, Pointed Dripstone, TNT, Respawn Anchor, Sculk Shrieker, Sculk Sensors. | 1 lava block surrounded by 63 stone blocks downsamples to `minecraft:lava`. |
| **INTERACTIVE / CRITICAL** | 5 | Containers (Chests, Barrels, Shulker Boxes, Hoppers), Workstations (Furnaces, Crafting Tables, Anvils, Enchanting Tables), Portals, Spawners, Trial Spawners, Redstone mechanisms, and **all Ores** (Coal, Iron, Diamond, Ancient Debris, etc.). | 2 diamond ores surrounded by 62 deepslate blocks downsample to `minecraft:diamond_ore`. |
| **LIQUID** | 4 | Water, Bubble Column. | Water surrounded by air/foliage downsamples to `minecraft:water`. |
| **FOLIAGE / PERMEABLE** | 3 | Leaves, Grass, Flowers, Crops, Saplings, Vines, Sugar Cane, Bamboo, Kelp, Seagrass, Cobwebs, Torches, Lanterns. | Vegetation is preserved over empty air. |
| **COMMON SOLID** | 2 | Stone, Deepslate, Granite, Diorite, Andesite, Tuff, Dirt, Grass Block, Sand, Gravel, Basalt, Blackstone, Wood/Planks, Bricks, Concrete, Glass. | Standard terrain and structural blocks; majority vote decides among solid blocks. |
| **AIR** | 1 | Air, Cave Air, Void Air. | Lowest priority; only selected if the cell is completely empty. |

#### Downsampling Procedure:
1. All blocks inside the $2^3$ or $4^3$ sub-volume are queried.
2. The highest priority rank present across all sub-blocks is determined (`highestRank`).
3. Candidate blocks are filtered strictly to those belonging to `highestRank`.
4. The most frequent block (mode) within that highest tier is selected.
5. In the event of a frequency tie within the same tier, alphabetical ordering of the block identifier is used for 100% deterministic outputs.

---

## 2. Command Reference

### `/observe` (and alias `/obs`)
Primary observation command for RL data collection.

```
/observe                      # Captures LOD 0 (16x16x16, stride 1)
/observe <0|1|2>              # Captures a specific LOD (0, 1, or 2)
/observe <0|1|2> <filename>   # Captures specific LOD to a custom filename
/observe all                  # Captures all 3 LODs (B0, B1, B2) in one file
/observe all <filename>       # Captures all 3 LODs to a custom filename
```

*Note: `/obs` is an alias and accepts identical arguments.*

#### File Destinations
Files are saved inside the Minecraft instance directory:
- Custom / Timestamped: `observations/obs_lod<LOD>_<timestamp>.json` (or `obs_all_<timestamp>.json`)
- Latest pointers (auto-overwritten for scripts):
  - `observations/obs_latest.json` (most recent capture of any kind)
  - `observations/obs_lod<LOD>_latest.json` (most recent capture for that specific LOD)

---

### `/squish` (Debug Command)
Arbitrary-sized box exporter retained for inspection and debugging.

```
/squish                       # Default 3x3x3 grid, LOD 1
/squish <size>                # <size>x<size>x<size> grid, LOD 1
/squish <size> <lod>          # Custom size and custom LOD downsampling factor
/squish <size> <lod> <file>   # Custom size, LOD, and filename
```
Outputs to `squish_output/`.

---

## 3. JSON Output Schema

### Single LOD Schema (`/observe 0`, `/observe 1`, or `/observe 2`)

```json
{
  "metadata": {
    "timestamp": "2026-11-08T04:15:00Z",
    "dimension": "minecraft:overworld",
    "gridSize": 16,
    "lod": 0,
    "stride": 1,
    "coordinateFrame": "relative_player_feet_egocentric",
    "downsampling": "priority_hierarchy",
    "worldCoverage": {
      "sizeX": 16,
      "sizeY": 16,
      "sizeZ": 16
    },
    "camera": {
      "rawYaw": 48.2,
      "rawPitch": 35.0,
      "snappedYaw": 0,
      "cardinalDirection": "SOUTH",
      "footIndex": 8,
      "playerForwardIndex": 2,
      "playerLateralIndex": 8
    },
    "playerWorldPos": { "x": 100, "y": 64, "z": 200 },
    "stats": {
      "totalCells": 4096,
      "totalWorldBlocks": 4096,
      "airCells": 2500,
      "nonAirCells": 1596,
      "blockCounts": {
        "minecraft:stone": 1200,
        "minecraft:dirt": 396,
        "minecraft:air": 2500
      }
    }
  },
  "grid": [
    ... 16x16x16 3D array: grid[lateral_x][vertical_y][forward_z] ...
    {
      "name": "minecraft:stone",
      "dx": -2,
      "dy": -1,
      "dz": 4
    }
  ]
}
```

### All LODs Schema (`/observe all`)

```json
{
  "metadata": {
    "timestamp": "2026-11-08T04:15:00Z",
    "dimension": "minecraft:overworld",
    "gridSize": 16,
    "coordinateFrame": "relative_player_feet_egocentric",
    "downsampling": "priority_hierarchy",
    "camera": { ... },
    "playerWorldPos": { ... }
  },
  "b0_lod0": [ ... 16x16x16 tensor ... ],
  "b1_lod1": [ ... 16x16x16 tensor ... ],
  "b2_lod2": [ ... 16x16x16 tensor ... ]
}
```

---

## 4. Python / PyTorch Integration Example

```python
import json
import numpy as np

def load_latest_observation(obs_path="observations/obs_latest.json"):
    with open(obs_path, "r") as f:
        data = json.load(f)

    meta = data["metadata"]
    print(f"Captured {meta['gridSize']}^3 observation facing {meta['camera']['cardinalDirection']}")
    print(f"Foot Level Index: gy={meta['camera']['footIndex']}, Player Forward: gz={meta['camera']['playerForwardIndex']}")

    # If captured via /observe all
    if "b0_lod0" in data:
        grid_b0 = data["b0_lod0"]
        grid_b1 = data["b1_lod1"]
        grid_b2 = data["b2_lod2"]
    else:
        grid_b0 = data["grid"]

    # Access cell at player position: [lateral=8][footIndex][forward=2]
    foot_idx = meta["camera"]["footIndex"]
    player_cell = grid_b0[8][foot_idx][2]
    print(f"Block at player's feet: {player_cell['name']} (dx={player_cell['dx']}, dy={player_cell['dy']}, dz={player_cell['dz']})")

    # Access 1 block directly ahead of the player: gz = 3
    block_ahead = grid_b0[8][foot_idx][3]
    print(f"Block directly in front: {block_ahead['name']} (dx={block_ahead['dx']}, dy={block_ahead['dy']}, dz={block_ahead['dz']})")

    return data

if __name__ == "__main__":
    load_latest_observation()
```

