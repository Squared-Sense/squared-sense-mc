# Squared Sense — World Model Repository Structure & Contracts

This document provides a single source of truth for the architecture, tensor contracts, data schemas, and evaluation protocols defined in the companion repository (`../Squared-Sense-World-Model/`). AI agents and offline training pipelines should consult this specification directly without needing to inspect the external repository.

---

## 1. Companion Repository Overview & Layout

The **Squared-Sense-World-Model** repository implements self-supervised 3D voxel representation learning (JEPA & MAE) using JAX and Flax NNX (`flax.nnx`), along with evaluation probes and a frozen-backbone PPO navigation policy.

### Directory Structure

```
Squared-Sense-World-Model/
├── docs/                               # Architectural and interface contracts
│   ├── CONTRACT.md                     # Phase 1 data and tensor schema contract (immutable)
│   ├── MODEL_CONTRACT.md               # Phase 2-3 JAX/Flax NNX neural module contract
│   ├── EVAL_CONTRACT.md                # Phase 5 evaluation, RL navigation & figures contract
│   ├── TRAIN_CONTRACT.md               # Training loops, loss formulations, and diagnostics
│   ├── ARCHITECTURE_PRETRAINING.md     # Mathematical formulation of JEPA and representation probes
│   └── PROGRESS.md                     # Completed phases and milestones
├── data/                               # Dataset definitions & schemas
│   ├── __init__.py
│   └── schema.py                       # Single source of truth for 16-channel enum, LOD geometry, constants
├── models/                             # JAX + Flax NNX neural networks (channel-last)
│   ├── __init__.py
│   ├── cnn3d.py                        # Siamese 3D ConvNeXt + SE + CoordConv3D (~0.471M params)
│   ├── transformer.py                  # MultiLODBackbone, Pos/Scale Emb, SpatialJEPA (~4.649M params)
│   ├── gating.py                       # Event-Driven Shell Gate (tau_gate = 0.05, zero trainable params)
│   ├── temporal_lstm.py                # Dual-LSTM + Query Cross-Attention (~0.725M params)
│   └── baselines.py                    # SingleLODJEPA (~4.65M) & Generative3DMAE (~5.13M)
├── train/                              # Representation learning training engines
│   ├── __init__.py
│   ├── checkpoint.py                   # Checkpoint serialization and restoring
│   ├── diagnostics.py                  # RankMe effective rank & representation health diagnostics
│   ├── losses.py                       # L1, Smooth L1, cosine similarity, VICReg, and probe losses
│   ├── train_spatial_jepa.py           # Stage 1A spatial JEPA training loop
│   ├── train_temporal_jepa.py          # Stage 1B temporal JEPA training loop
│   └── train_baselines.py              # Baseline comparison training
├── eval/                               # Evaluation benchmarks & RL navigation
│   ├── __init__.py
│   ├── sim_env.py                      # Deterministic 3D Voxel Navigation Simulator (VoxelNavEnv)
│   ├── train_ppo_nav.py                # Frozen-Backbone PPO Policy (~50K param ActorCriticHead)
│   ├── eval_navigation.py              # Course evaluation, SPL, SR%, latency & gating profiling
│   └── probe_suite.py                  # Linear / MLP probe suite (goal direction, hazard, velocity, memory)
├── paper_figs/                         # Springer LNEE-compliant vector publication figures
│   └── make_figures.py                 # Generates Fig 1-4 vector plots (.pdf / .png)
└── tests/                              # Pytest test suite covering all modules
```

---

## 2. 16-Channel Voxel Schema Specification

The world model operates exclusively on a continuous 16-channel volumetric voxel tensor.
- **Disk Representation**: `uint8` in $[0\dots 255]$.
- **Neural Input**: `float32` in $[0.0\dots 1.0]$ via $x_{\text{norm}} = \frac{x_{\text{uint8}}}{255.0}$.
- **Layout**: Strictly channel-last: `(..., X, Y, Z, C)` where $C = 16$.

### Channel Definitions (`data/schema.py`)

| Index | Channel Name | Semantic Meaning | Pooling Strategy across LODs |
|---|---|---|---|
| **0** | `TERRAIN` (`SOLID`) | Solid opaque walkable terrain (`is_solid == True`). | `AVG` pool |
| **1** | `AIR` | Passable empty space (`passability == 1.0` and not liquid). | `AVG` pool |
| **2** | `HAZARD` | Lava, fire, magma, sweet berry bushes (`hazard >= 0.5`). | `MAX` pool (salience) |
| **3** | `LIQUID` (`WATER`) | Water and fluid blocks. | `AVG` pool |
| **4** | `CLIMBABLE` | Ladders, vines, stairs (vertical stepping aids). | `MAX` pool (salience) |
| **5** | `FOLIAGE` | Grass, flowers, crops, saplings, leaves (fragile/breakable). | `AVG` pool |
| **6** | `WORLD_Y` | Normalized world height: $\frac{y_{\text{world}} + 64}{384} \in [0.0, 1.0]$. | `AVG` pool |
| **7** | `COLLISION_HEIGHT` | Collision height: $0.0$ (air), $0.5$ (slabs), $1.0$ (full block/fence). | `MAX` pool (salience) |
| **8** | `WAYPOINT` (`GOAL`) | Active goal beacon indicator ($1.0$ at target voxel, $0.0$ elsewhere). | `MAX` pool (salience) |
| **9** | `RESOURCE` | Interactive blocks, crafting tables, chests, ores. | `MAX` pool (salience) |
| **10** | `WALKABLE_SURFACE` | Top of solid block having at least 2 blocks of air clearance above. | `MAX` pool (salience) |
| **11** | `LIGHT_LEVEL` | Combined illumination: $\max(\text{sky}, \text{block}) / 15.0 \in [0.0, 1.0]$. | `AVG` pool |
| **12** | `LETHAL_DROP` (`CLIFF`) | Dangerous drop: $\ge 3$ consecutive air blocks directly below feet. | `MAX` pool (salience) |
| **13** | `SURFACE_SHELL` | Surface boundary: solid block exposed directly to air (not buried). | `MAX` pool (salience) |
| **14** | `BLOCK_HARDNESS` | Normalized block mining resistance / hardness $\in [0.0, 1.0]$. | `AVG` pool |
| **15** | `OUT_OF_BOUNDS` (`VOID`) | World boundary padding mask or void space. | `MAX` pool |

### Pooling Partition Sets

Downsampling from high resolution to coarser LODs uses a strict disjoint and exhaustive partition:
- **`MAX_POOL_CHANNELS`** (salient features): `(2, 4, 7, 8, 9, 10, 12, 13, 15)`
- **`AVG_POOL_CHANNELS`** (volumetric averages): `(0, 1, 3, 5, 6, 11, 14)`

---

## 3. Concentric Multi-LOD Shell Geometry

Perception extracts 3 concentric egocentric shell grids. Every shell produces an invariant output shape of `(16, 16, 16, 16)` = $(X, Y, Z, C)$.

| Shell | Scale | Physical Span | Forward Reach ($+Z$) | Rear Reach ($-Z$) | Lateral Span ($\pm X$) | Base Height ($Y$) |
|---|---|---|---|---|---|---|
| **LOD-0** | $1\times$ (1m) | $16\times 16\times 16$ blocks | $+12$ blocks | $-4$ blocks | $\pm 8$ blocks | $-4\dots +12$ blocks |
| **LOD-1** | $2\times$ (2m) | $32\times 32\times 32$ blocks | $+24$ blocks | $-8$ blocks | $\pm 16$ blocks | $-8\dots +24$ blocks |
| **LOD-2** | $4\times$ (4m) | $64\times 64\times 64$ blocks | $+48$ blocks | $-16$ blocks | $\pm 32$ blocks | $-16\dots +48$ blocks |

### Geometric Rules & Invariants
- **Player Foot Anchor**: In local coordinates for LOD-0, the player's feet sit at $(X = 8, Z = 4, Y = 4)$ relative to the bounding box minimum $(x_{\min}, z_{\min}, y_{\min})$.
- **Pitch Dynamic Shift**: The vertical sampling box shifts with camera pitch angle $\phi$:
  $$\text{shift}_y = \text{round}(4 \cdot \sin(\phi)) \times \text{scale}$$
- **Cardinal Snapping**: Player yaw is snapped to cardinal headings ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) with rotational hysteresis deadband to eliminate orientation jitter.

---

## 4. Trajectory & Dataset File Contracts

### Option B Episode Trajectories (5 Hz Time-Series)
For self-supervised pretraining and offline reinforcement learning:
- `shells`: `(T, 3, 16, 16, 16, 16)` `uint8` on disk, normalized to `float32` in model.
- `state`: `(T, 12)` `float32`:
  $$[v_x, v_y, v_z, \Delta x, \Delta y, \Delta z, \sin(\text{yaw}), \cos(\text{yaw}), \text{pitch}, \text{health}, \text{food}, \text{grounded}]$$
- `action`: `(T,)` `int64` (discrete movement primitive index).
- `meta`: `(T, 6)` `float32`:
  $$[x_{\text{world}}, y_{\text{world}}, z_{\text{world}}, \text{goal}_{\Delta x}, \text{goal}_{\Delta y}, \text{goal}_{\Delta z}]$$
- **Dataset Partition**: $70\%$ train, $15\%$ validation, $15\%$ test split strictly by episode ID (never frame shuffle).

### Option A Full-Course Arena Volumes
For static navigation evaluation courses:
- Arena volume array: `(16, 128, 64, 128)` `uint8` (`C, X, Y, Z`).
- Waypoint list JSON: `courses/<course_name>_waypoints.json` containing ordered 3D integer coordinates `[[x, y, z], ...]`.

---

## 5. Neural Architecture & Parameter Budgets (JAX / Flax NNX)

All models adhere to strict parameter budgets and use channel-last tensor layouts:

```
Voxel Shells Input: (B, 3, 16, 16, 16, 16) float32
      │
      ▼
Siamese 3D ConvNeXt Encoder (models/cnn3d.py)
  ├─ CoordConv3D (normalized [u, v, w] grid -> 19 channels)
  ├─ Stem Conv3D: (19 -> 64, stride 2) -> (N, 8, 8, 8, 64)
  ├─ Stage 1: 2x ConvNeXt3DSEBlock(64) + Squeeze-and-Excitation
  ├─ Downsample Conv3D: (64 -> 128, stride 2) -> (N, 4, 4, 4, 128)
  ├─ Stage 2: 2x ConvNeXt3DSEBlock(128) + Squeeze-and-Excitation
  └─ Projection: 128 -> 256 -> (N, 4, 4, 4, 256) -> flatten to (N, 64, 256)
      │
      ▼
Tokens: (B, 192, 256) (3 LODs × 64 tokens per shell, D=256)
      │
      ▼
MultiLODBackbone Transformer (models/transformer.py)
  ├─ Learned 3D Coordinate + Scale Positional Embeddings (17,152 params)
  ├─ 4-Layer Pre-LN Multi-Head Self-Attention (dim=256, heads=8, mlp=4x) (3.16M params)
  └─ Context Representations: (B, 192, 256)
```

### Module Parameter Counts
- `models/cnn3d.py`: Siamese 3D ConvNeXt + SE + CoordConv: $\approx 470,560$ params ($0.471\text{M}$).
- `models/transformer.py`: Core Transformer Backbone: $\approx 3,647,264$ params ($3.647\text{M}$); SpatialJEPA total: $\approx 4.649\text{M}$ params.
- `models/gating.py`: Event-Driven Shell Gating ($\tau_{\text{gate}} = 0.05$): $0$ trainable parameters (stateful cache).
- `models/temporal_lstm.py`: Dual-LSTM + Query Cross-Attention: $\approx 724,864$ params ($0.725\text{M}$).
- `eval/train_ppo_nav.py`: Frozen-backbone ActorCriticHead: $\approx 50,000$ params ($0.05\text{M}$).

---

## 6. RL Navigation Simulator & Policy Contract

### 12-Action Discrete Action Space (`eval/sim_env.py`)

| Action ID | Action Name | Mathematical Translation / Transformation |
|---|---|---|
| **0** | `Forward` | Step forward: $\Delta x = \text{round}(\sin(\text{yaw})), \Delta z = \text{round}(\cos(\text{yaw}))$. |
| **1** | `Forward-Left` | Diagonal step forward-left: $\Delta x = \text{round}(\vec{F}_x + \vec{L}_x), \Delta z = \text{round}(\vec{F}_z + \vec{L}_z)$. |
| **2** | `Forward-Right` | Diagonal step forward-right: $\Delta x = \text{round}(\vec{F}_x + \vec{R}_x), \Delta z = \text{round}(\vec{F}_z + \vec{R}_z)$. |
| **3** | `Strafe-Left` | Lateral left step: $\Delta x = \text{round}(\vec{L}_x), \Delta z = \text{round}(\vec{L}_z)$. |
| **4** | `Strafe-Right` | Lateral right step: $\Delta x = \text{round}(\vec{R}_x), \Delta z = \text{round}(\vec{R}_z)$. |
| **5** | `Backward` | Step backward: $\Delta x = \text{round}(-\vec{F}_x), \Delta z = \text{round}(-\vec{F}_z)$. |
| **6** | `Turn-Left` | Rotate yaw $-45^\circ$: $\text{yaw} = (\text{yaw} - 45^\circ) \bmod 360^\circ$. |
| **7** | `Turn-Right` | Rotate yaw $+45^\circ$: $\text{yaw} = (\text{yaw} + 45^\circ) \bmod 360^\circ$. |
| **8** | `Step-Up-Forward` | Forward step with explicit 1-block vertical climb attempt. |
| **9** | `Pitch-Up` | Rotate head pitch $+15^\circ$: $\min(90^\circ, \text{pitch} + 15^\circ)$. |
| **10** | `Pitch-Down` | Rotate head pitch $-15^\circ$: $\max(-90^\circ, \text{pitch} - 15^\circ)$. |
| **11** | `Stay / Idle` | No movement or rotation. |

### Physics Kernel & Termination Rules
1. **1-Block Step-Up**: Permitted when target block at foot level is solid, but clearance exists at height $+1$ and $+2$.
2. **Safe Drops**: Drops $\le 2$ blocks vertically are completely safe.
3. **Lethal Falls**: Drops $\ge 3$ blocks trigger fall penalty ($-3.0$) and episode termination (`collision = True`).
4. **Hazard Contact**: Stepping into a voxel with $\text{hazard} \ge 128$ (lava/fire) triggers hazard penalty ($-5.0$) and termination.
5. **Wall Bumps**: Attempting to walk into solid walls or insufficient headroom triggers a bump penalty ($-0.1$).

### Potential-Based Reward Shaping
- **Waypoint Arrival**: $+10.0$ upon reaching within $1.5$ blocks of active waypoint.
- **Progress Reward**: $+0.25 \times (d_{t-1} - d_t)$ (potential-based shaping to active goal).
- **Step Cost**: $-0.02$ per step.
- **Termination Penalties**: $-5.0$ (hazard), $-3.0$ (fatal fall $\ge 3$ blocks).

---

## 7. Interoperability Adapter (`Squared Sence MC` $\leftrightarrow$ World Model)

To feed observations from `Squared Sence MC` into the world model or use world model checkpoints for navigation:

### Channel Derivation Mapping

```
Squared Sence MC Palette & Physicals          World Model 16-Channel Schema
─────────────────────────────────────          ─────────────────────────────
is_solid == True                      ───────► ch 0: TERRAIN
passability == 1.0 (not liquid)       ───────► ch 1: AIR
hazard >= 0.5 (lava/fire/magma)       ───────► ch 2: HAZARD
token in WATER_BLOCKS                 ───────► ch 3: LIQUID
token in CLIMBABLE_BLOCKS             ───────► ch 4: CLIMBABLE
token in FOLIAGE_BLOCKS               ───────► ch 5: FOLIAGE
(world_y + 64) / 384.0                ───────► ch 6: WORLD_Y
1.0 (solid) / 0.5 (slab) / 0.0 (air)  ───────► ch 7: COLLISION_HEIGHT
1.0 at relative goal voxel            ───────► ch 8: WAYPOINT
token in RESOURCE_BLOCKS              ───────► ch 9: RESOURCE
solid floor with >= 2 air above       ───────► ch 10: WALKABLE_SURFACE
normalized incident light [0..1]      ───────► ch 11: LIGHT_LEVEL
>= 3 air directly below foot          ───────► ch 12: LETHAL_DROP
visibility > 0.0 (exposed shell)      ───────► ch 13: SURFACE_SHELL
hardness in [0.0, 1.0]                ───────► ch 14: BLOCK_HARDNESS
voxel out of loaded region / void     ───────► ch 15: OUT_OF_BOUNDS
```

### Action Space Interoperability
- **Standard Navigation (4 Actions)**: Forward (0), Backward (1), Left (2), Right (3). Maps directly to World Model actions 0, 5, 3, 4 with auto-step-up.
- **Cardinal Navigation (6 Actions)**: Adds Turn-Left (4) and Turn-Right (5) (90° turns). Maps to 2x 45° turns in World Model actions 6, 7.
- **Full Navigation (12 Actions)**: Exact 1-to-1 parity with `VoxelNavEnv`.

