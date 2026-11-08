# Observation Specification & Commands

This document specifies the data formats, command interfaces, and integration patterns for consuming Squared Sense MC observations in Python / PyTorch RL pipelines.

---

## 1. LOD (Level of Detail) System

Observations use a fixed $16\times 16\times 16$ tensor volume across 3 hierarchical LOD levels:

| Level | Block Tensor ($B_n$) | Entity Tensor ($E_n$) | Stride (Scale) | Physical World Coverage | Sub-Volume per Cell |
|---|---|---|---|---|---|
| **LOD 0** | $B_0$ ($16\times 16\times 16$) | $E_0$ ($16\times 16\times 16$) | $1$ block | $16\times 16\times 16$ meters | $1\times 1\times 1$ ($1$ block) |
| **LOD 1** | $B_1$ ($16\times 16\times 16$) | $E_1$ ($16\times 16\times 16$) | $2$ blocks | $32\times 32\times 32$ meters | $2\times 2\times 2$ ($8$ blocks) |
| **LOD 2** | $B_2$ ($16\times 16\times 16$) | $E_2$ ($16\times 16\times 16$) | $4$ blocks | $64\times 64\times 64$ meters | $4\times 4\times 4$ ($64$ blocks) |

Each entity tensor $E_n$ matches the spatial bounds, egocentric rotation, pitch translation, and cell resolution of block tensor $B_n$ 1:1, carrying 3 aggregated channels per cell: `entityCount`, `isHostile` (max-pooled), and `totalHealth`.

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
/observe vocab                # Exports canonical block_vocab.json (Task 9)
/observe stream start [t]     # Starts real-time MMF streaming every [t] ticks (default: 2 ticks / 10 Hz)
/observe stream stop          # Stops real-time MMF streaming
/observe stream status        # Checks current MMF streaming state, FPS, and frame count
```

*Note: `/obs` is an alias and accepts identical arguments (e.g., `/obs stream start`, `/obs vocab`, `/obs all`).*

#### File Destinations
Files are saved inside the Minecraft instance directory:
- Custom / Timestamped: `observations/obs_lod<LOD>_<timestamp>.json` (or `obs_all_<timestamp>.json`)
- Canonical Vocabulary: `observations/block_vocab.json` (auto-generated or exported via `/observe vocab`)
- Shared Memory MMF: `/dev/shm/squared_sense_mc.bin` on Linux, `%TEMP%\squared_sense_mc.bin` on Windows
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
    "vocabSize": 882,
    "channels": [
      "id",
      "name",
      "dx",
      "dy",
      "dz",
      "occupancy",
      "passability",
      "hazard",
      "hardness",
      "lightLevel",
      "fluidFlow"
    ],
    "entityChannels": [
      "entityCount",
      "isHostile",
      "totalHealth"
    ],
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
    },
    "entityStats": {
      "totalEntities": 1,
      "hostileEntities": 1,
      "passiveEntities": 0,
      "countsByType": {
        "minecraft:zombie": 1
      }
    }
  },
  "grid": [
    ... 16x16x16 3D array: grid[lateral_x][vertical_y][forward_z] ...
    {
      "id": 782,
      "name": "minecraft:stone",
      "dx": -2,
      "dy": -1,
      "dz": 4,
      "properties": {
        "occupancy": 1.0,
        "passability": 0.0,
        "hazard": 0.0,
        "hardness": 1.5,
        "lightLevel": 0.0,
        "fluidFlow": [0.0, 0.0]
      }
    }
  ],
  "entities": [
    ... 16x16x16 3D array: entities[lateral_x][vertical_y][forward_z] ...
    {
      "dx": 0,
      "dy": 0,
      "dz": 3,
      "entityCount": 1,
      "isHostile": 1.0,
      "totalHealth": 20.0,
      "entities": [
        {
          "type": "minecraft:zombie",
          "health": 20.0,
          "hostile": true,
          "localX": 0.12,
          "localY": 0.0,
          "localZ": 3.45
        }
      ]
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
    "entityChannels": ["entityCount", "isHostile", "totalHealth"],
    "entityStats": {
      "lod0": { "totalEntities": 1, "hostileEntities": 1, "passiveEntities": 0, ... },
      "lod1": { "totalEntities": 3, "hostileEntities": 2, "passiveEntities": 1, ... },
      "lod2": { "totalEntities": 8, "hostileEntities": 5, "passiveEntities": 3, ... }
    },
    "camera": { ... },
    "playerWorldPos": { ... }
  },
  "b0_lod0": [ ... 16x16x16 block tensor ... ],
  "b1_lod1": [ ... 16x16x16 block tensor ... ],
  "b2_lod2": [ ... 16x16x16 block tensor ... ],
  "e0_lod0": [ ... 16x16x16 entity tensor ... ],
  "e1_lod1": [ ... 16x16x16 entity tensor ... ],
  "e2_lod2": [ ... 16x16x16 entity tensor ... ]
}
```

---

## 4. Python / PyTorch Integration Example

```python
import json
import numpy as np

def load_latest_observation(obs_path="observations/obs_latest.json", vocab_path="observations/block_vocab.json"):
    with open(obs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Optional: load canonical vocabulary
    try:
        with open(vocab_path, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
            print(f"Loaded block vocabulary: {vocab['vocabSize']} tokens")
    except FileNotFoundError:
        vocab = None

    meta = data["metadata"]
    print(f"Captured {meta['gridSize']}^3 observation facing {meta['camera']['cardinalDirection']}")
    print(f"Foot Level Index: gy={meta['camera']['footIndex']}, Player Forward: gz={meta['camera']['playerForwardIndex']}")
    print(f"Vocab Size: {meta.get('vocabSize')}, Channels: {meta.get('channels')}")

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
    print(f"Block at player's feet: {player_cell['name']} (ID={player_cell.get('id')}, dx={player_cell['dx']}, dy={player_cell['dy']}, dz={player_cell['dz']})")
    print(f"  Physical: occupancy={player_cell['properties']['occupancy']}, passability={player_cell['properties']['passability']}")

    # Access 1 block directly ahead of the player: gz = 3
    block_ahead = grid_b0[8][foot_idx][3]
    print(f"Block directly in front: {block_ahead['name']} (ID={block_ahead.get('id')}, dx={block_ahead['dx']}, dy={block_ahead['dy']}, dz={block_ahead['dz']})")

    # Access entities in front of the player (Task 10)
    entities_grid = data.get("e0_lod0", data.get("entities"))
    if entities_grid:
        ahead_entities = entities_grid[8][foot_idx][3]
        print(f"Entities in front: count={ahead_entities['entityCount']}, hostile={ahead_entities['isHostile']}, health={ahead_entities['totalHealth']}")

    return data

if __name__ == "__main__":
    load_latest_observation()
```

