# Entity Observations ($E_0, E_1, E_2$) — Task 10

## 1. Executive Summary & Design Rationale

While block grids ($B_0, B_1, B_2$) provide static environmental terrain and physical affordances, an autonomous agent in Minecraft must perceive and react to **dynamic agents and mobs** (hostile monsters, passive animals, items, and NPCs).

Rather than forcing sparse entity coordinates into a separate flat list that loses spatial locality, Squared Sense MC constructs **matching 3D entity tensor grids ($E_0, E_1, E_2$)**:

1. **Exact 1:1 Spatial Registration**:
   - Each entity grid $E_n$ is a $16\times 16\times 16$ tensor sharing the exact same world bounding box, egocentric rotation, vertical pitch translation, and LOD stride as block grid $B_n$.
   - A zombie standing on block $(gx, gy, gz)$ in $B_n$ occupies the identical voxel index $(gx, gy, gz)$ in $E_n$.
2. **Standard 3D CNN Input**:
   - Since $E_n$ and $B_n$ share identical spatial dimensions ($16\times 16\times 16$), downstream neural networks can concatenate them directly along the channel dimension or process them through dual 3D convolutional encoders.
3. **Multi-Scale Awareness**:
   - $E_0$ (stride 1) captures precision combat positioning within $16\times 16\times 16$ meters.
   - $E_1$ (stride 2) detects approaching mobs within $32\times 32\times 32$ meters.
   - $E_2$ (stride 4) spots distant threats and wandering animals within $64\times 64\times 64$ meters.

---

## 2. Channels per Entity Voxel

Each voxel $(gx, gy, gz)$ in $E_n$ aggregates the following 3 feature channels:

| Channel | Key | Type | Value Range | Semantic Purpose |
|---|---|---|---|---|
| **0** | `entityCount` | Integer / Float | $[0, \infty)$ | Number of entities occupying the cell's volume ($s \times s \times s$ meters). Usually 0 or 1, occasionally $\ge 2$ when mobs crowd together. |
| **1** | `isHostile` | Float | $\{0.0, 1.0\}$ | **Max-pooled hostility flag**. Evaluates to $1.0$ if *any* entity in the cell is actively hostile towards the player; else $0.0$. |
| **2** | `totalHealth` | Float | $[0.0, \infty)$ | Sum of the current health of all living entities in the cell. Provides the agent with damage and threat assessment. Non-living entities contribute $0.0$. |

In debug JSON representations, non-empty cells (`entityCount > 0`) additionally contain an `"entities"` array detailing the entity type (`minecraft:zombie`), precise health, and continuous local offsets (`localX`, `localY`, `localZ`).

---

## 3. Hostility Classification Logic

Classification is handled by [`EntityClassifier.java`](file:///c:/Users/Krishna/Projects/Squared%20Sence%20MC/src/client/java/org/sqsense/squared_sense_mc/client/observation/entity/EntityClassifier.java):

```
                       [Entity in Query Box]
                                 │
                   Is it local player or dead?
                           ┌─────┴─────┐
                         Yes          No
                          │            │
                      [Exclude]   [Evaluate]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        instanceof Monster?                   instanceof Angerable?
          (Zombie, Skeleton, Creeper,          (Zombified Piglin, Wolf,
           Enderman, Slime, Blaze...)           Iron Golem, Piglin...)
                    │                                     │
                 Hostile                             Has anger timer?
                                                    ┌─────┴─────┐
                                                  Yes          No
                                                   │            │
                                                Hostile      Passive
```

### Supported Categories:
1. **Inherently Hostile (`Monster`)**:
   - Evaluates to `isHostile = 1.0`.
   - Includes Zombies, Skeletons, Creepers, Spiders, Cave Spiders, Endermen, Witches, Slimes, Magma Cubes, Ghasts, Blazes, Phantoms, Pillagers, Vindicators, Evokers, Ravagers, Wardens, Shulkers, Hoglins, Piglin Brutes, Guardians, and 1.21 Breeze / Bogged.
2. **Angerable Neutral Mobs (`Angerable`)**:
   - If `angerable.hasAngerTime()` is true, evaluates to `isHostile = 1.0`.
   - If calm/unprovoked, evaluates to `isHostile = 0.0`.
   - Includes Wolves, Zombified Piglins, Piglins, Iron Golems, Polar Bears, and Bees.
3. **Passive Animals & NPCs**:
   - Evaluates to `isHostile = 0.0`.
   - Includes Cows, Pigs, Sheep, Chickens, Horses, Villagers, and Allays.
4. **Living Health Extraction**:
   - Extracted via `LivingEntity.getHealth()`, synchronized from client data trackers.
   - Non-living entities (minecarts, boats, arrows, dropped items) return health `0.0`.

---

## 4. Coordinate Projection Math

Given an entity at world position $(X_e, Y_e, Z_e)$ and the observing player at foot position $(X_p, Y_p, Z_p)$:

### Step 1: Discrete Block Coordinates
Minecraft's discrete block coordinates are derived by flooring:
$$\Delta X_{world} = \text{entity.getBlockX()} - X_p$$
$$\Delta Y_{world} = \text{entity.getBlockY()} - Y_p$$
$$\Delta Z_{world} = \text{entity.getBlockZ()} - Z_p$$

### Step 2: Inverse Orthonormal Rotation
Using the snapped cardinal direction's right vector $\vec{R} = (R_x, R_z)$ and forward vector $\vec{F} = (F_x, F_z)$, the world horizontal delta is projected into the egocentric local reference frame:
$$x_{local} = \Delta X_{world} \cdot R_x + \Delta Z_{world} \cdot R_z$$
$$z_{local} = \Delta X_{world} \cdot F_x + \Delta Z_{world} \cdot F_z$$
$$y_{local} = \Delta Y_{world}$$

### Step 3: Stride Partitioning with `Math.floorDiv`
Each cell $(gx, gy, gz)$ in an LOD with stride $s$ partitions the continuous local coordinate space into bins of width $s$:
$$gx = \left\lfloor \frac{x_{local}}{s} \right\rfloor + \text{PLAYER\_LATERAL\_INDEX} \quad (\text{index } 8)$$
$$gy = \left\lfloor \frac{y_{local}}{s} \right\rfloor + \text{footIndex} \quad (\text{pitch-dependent } [1..12])$$
$$gz = \left\lfloor \frac{z_{local}}{s} \right\rfloor + \text{PLAYER\_FORWARD\_INDEX} \quad (\text{index } 2)$$

Using Java's `Math.floorDiv` guarantees mathematically consistent rounding across negative coordinates, ensuring contiguous tiling without boundaries or overlapping cells.

---

## 5. JSON Output Schema

### Single LOD Schema (`/observe <0|1|2>`)
Observations now include both the `"grid"` (blocks) and `"entities"` ($E_n$):

```json
{
  "metadata": {
    "gridSize": 16,
    "lod": 0,
    "stride": 1,
    "entityChannels": ["entityCount", "isHostile", "totalHealth"],
    "entityStats": {
      "totalEntities": 2,
      "hostileEntities": 1,
      "passiveEntities": 1,
      "countsByType": {
        "minecraft:zombie": 1,
        "minecraft:cow": 1
      }
    }
  },
  "grid": [ ... 16x16x16 block voxels ... ],
  "entities": [
    ... 16x16x16 entity voxels: entities[gx][gy][gz] ...
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
When capturing all LODs, block grids ($B_0, B_1, B_2$) and entity grids ($E_0, E_1, E_2$) are emitted in parallel:
- `b0_lod0`, `b1_lod1`, `b2_lod2`
- `e0_lod0`, `e1_lod1`, `e2_lod2`

---

## 6. PyTorch / RL Integration Example

In PyTorch, the block channels and entity channels can be merged directly into a single multi-channel 3D volume:

```python
import json
import torch
import torch.nn as nn

# Load observation
with open("observations/obs_latest.json", "r", encoding="utf-8") as f:
    obs = json.load(f)

# Extract entity channels into tensor of shape (3, 16, 16, 16)
entity_tensor = torch.zeros((3, 16, 16, 16), dtype=torch.float32)
entities_grid = obs["entities"]

for gx in range(16):
    for gy in range(16):
        for gz in range(16):
            cell = entities_grid[gx][gy][gz]
            if cell["entityCount"] > 0:
                entity_tensor[0, gx, gy, gz] = cell["entityCount"]
                entity_tensor[1, gx, gy, gz] = cell["isHostile"]
                entity_tensor[2, gx, gy, gz] = cell["totalHealth"]

# Dual-Stream 3D CNN Backbone
class DualStreamPerception(nn.Module):
    def __init__(self, block_channels: int = 15, entity_channels: int = 3):
        super().__init__()
        # Early-fusion 3D convolutional encoder
        total_channels = block_channels + entity_channels # 18 channels
        
        self.encoder = nn.Sequential(
            nn.Conv3d(total_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # -> 8x8x8
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # -> 4x4x4
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, block_feat: torch.Tensor, entity_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate block features and entity features along channel dim
        x = torch.cat([block_feat, entity_feat], dim=1) # (B, 18, 16, 16, 16)
        return self.encoder(x)
```

