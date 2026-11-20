# MCA Dataset & Pathfinding Pretraining Plan

## Overview

This document outlines the offline data generation and pathfinding pretraining pipeline built on top of Chunky-generated Minecraft worlds. By baking `.mca` region files into memory-mapped NumPy arrays and serving observations in the exact schema expected by the model and 3D visualizer, we can run high-throughput parallel training and inspect single-episode inference runs on the fly.

## 1. Key Engineering & RL Design Decisions

* **One-Time `.mca` Baking (No Live NBT Parsing):**
  A single `.mca` file covers $512 \times 512$ blocks ($32 \times 32$ chunks). Decompressing zlib/NBT chunk data on the fly during RL steps will bottleneck training. All `.mca` regions are converted once into flat, read-only NumPy memory-mapped arrays (`.memmap` / `.npy`), enabling microsecond slicing and shared zero-copy access across parallel RL workers.

* **Target Visibility & Reachability Guarantee (Bounded BFS inside $B_2$):**
  To avoid out-of-sight or unreachable goals under a static viewing orientation, all candidate target blocks are sampled via a bounded 3D Breadth-First Search (BFS) clamped strictly within the spatial bounding box of the largest LOD ($B_2$) relative to the agent's spawn position. This guarantees every target is both **visible in at least $B_2$** and **100% traversable** under the agent's physics rules, while also providing the true optimal path length $L_{\text{BFS}}$.

* **Observation State vs. Visualizer Telemetry:**
  * **Model Observation State:** Contains the standard LOD observation schema + relative goal vector (`goal_delta_xyz`, `goal_dist`) and optionally `prev_action`. Training loss is *not* fed into the model as an input (since it does not exist at inference time).
  * **Visualizer Telemetry Header:** Training/checkpoint losses, Critic value estimate $V(s)$, action probabilities, and target/path overlays are attached separately in the IPC stream for the visualizer's inspection panel.

* **Potential-Based Reward Shaping:**
  * **Progress Reward:** $+\alpha \cdot (d_{t-1} - d_t)$ (rewards closing distance to the target, penalizes moving away, and prevents oscillation exploits).
  * **Step Penalty:** $-0.02$ per step to discourage standing still.
  * **Blocked / Illegal Move Penalty:** $-0.05$ (or hard action masking so illegal moves cannot be sampled).
  * **Goal Reached Bonus:** $+15.0$ upon reaching within $\le 1$ block of the target.
  * **Fatal Hazard / Fall Penalty:** $-10.0$ for touching lava/fire or dropping $> 5$ blocks.
  * **Episode Truncation:** Timeout after $T_{\max} = 3 \times L_{\text{BFS}}$ steps.

## 2. Phased Implementation Checklist

### Phase 1: World Generation & Fast Binary Baking
*Turn Chunky-generated Minecraft worlds into instant-access NumPy terrain.*

* [x] **Generate Worlds with Chunky:** Pre-generate 2–3 diverse seeds (start with $4,096 \times 4,096$ blocks per seed = $64$ `.mca` files each; scale up to $10\text{k} \times 10\text{k}$ once the pipeline is verified). *(Code pipeline ready and tested against real MCA region saves)*
* [x] **Sync Shared Block Palette & Static Properties:** Reference the same block ID palette (`0 = AIR`, `1 = MASKED`), priority tiers, and static physical property LUTs (`passability`, `hazard`, `hardness`) used by the live mod.
* [x] **Build the `.mca` $\rightarrow$ Memmap Baker Script:** Use `amulet-core` (or `anvil-parser`) in Python to parse each region once and dump `(X, Y, Z)` integer block IDs (and chunk light levels, if present) into flat `.memmap` files on disk.
* [x] **Precompute Terrain Spawn Index:** Scan the baked arrays once to save lists of valid spawn coordinates (2-block-tall air space above a solid, non-hazard block) split by terrain type:
    * [x] **60% Surface:** Standard terrain at the heightmap surface (plains, forests, rivers, hills).
    * [x] **25% Subterranean:** Underground air pockets ($Y < 50$) inside caves and tunnels.
    * [x] **15% High-Elevation / Steep:** Surface coordinates with sharp local heightmap gradients (cliffs, ravines, mountain ridges).

### Phase 2: NumPy LOD Extractor (Unified Schema)
*Extract LOD blobs on the fly from the memmap in the exact schema used by the mod and visualizer.*

* [x] **Fast Coordinate Slicer:** Given `(player_x, player_y, player_z)` and a static cardinal orientation (for v1), slice the $B_2$ world volume directly from the `.memmap` array.
* [x] **Vectorized LOD Downsampler:** Compute $B_0, B_1, B_2$ using NumPy LUT lookups, priority pooling for block IDs, and physical channel aggregation.
* [x] **Static Placeholders for Uncomputed Fields:** Zero-fill dynamic fields not needed for static terrain/pathfinding (e.g., entity grids, facing entity, complex game-computed fluid flow) so the output tensor schema stays 100% compatible with the model and visualizer.
* [x] **Schema Verification:** Stream a test extraction from Python directly into the 3D visualizer to confirm parity with the mod's output.

### Phase 3: Offline Pathfinding RL Environment (Gymnasium)
*Build the fast simulation loop with traversable, visible targets.*

* [x] **Deterministic Movement Physics Kernel:**
    * [x] Define the discrete movement action space (legal horizontal moves; optional action masking for blocked directions).
    * [x] Enforce the $1 \times 2 \times 1$ player hitbox: allow stepping up $1$ block when headroom is clear, and resolve gravity by falling straight down until hitting solid ground while tracking fall distance.
    * [x] Flag fatal outcomes: fall distance $> 5$ blocks, landing in lava/fire, or suffocation.
* [x] **Bounded BFS Target Generator (Strictly Inside $B_2$):**
    * [x] On `reset()`, pick a spawn point from the Phase 1 index following the 60/25/15 terrain split.
    * [x] Run a fast 3D BFS outward from spawn using the movement kernel, clamping the search strictly within the spatial bounds of $B_2$.
    * [x] Sample a target block from the reachable frontier matching the current curriculum distance and store $L_{\text{BFS}}$.
* [x] **Observation & Reward Wrapper:**
    * [x] Return `{LOD_blob, metadata, goal_delta_xyz, goal_dist, prev_action}`.
    * [x] Implement the potential-based progress reward, step penalty, hazard/fall penalty, goal bonus, and $3 \times L_{\text{BFS}}$ timeout.
* [x] **Distance Curriculum Scheduler:** Gradually expand target sampling distance as success rate improves (e.g., inside $B_0$ $\rightarrow$ inside $B_1$ $\rightarrow$ full $B_2$ horizon).

### Phase 4: Parallel Training Pipeline
*Run fast, parallelized training jobs directly against the memmap worlds.*

* [ ] **Vectorized Parallel Workers:** Wrap the Gymnasium environment in `AsyncVectorEnv` / `SubprocVecEnv` (8–32 workers) all reading from the same shared read-only `.memmap` files with minimal RAM overhead.
* [ ] **Training Loop Integration:** Connect the parallel environments to your RL training script (e.g., PPO) and optional masked-voxel spatial pretraining sampling random slices from the same memmap files.
* [ ] **Checkpointing & Logging:** Save periodic checkpoints alongside training metrics (policy/value loss, entropy, episode success rate, and path efficiency ratio $\frac{\text{actual steps}}{L_{\text{BFS}}}$).

### Phase 5: Single-Inference Mode & Visualizer Inspection Panel
*Inspect trained checkpoints step-by-step in real time.*

* [ ] **IPC Telemetry Extension:** Extend the data blob sent to the visualizer with an optional **Inference Telemetry Header**:
  * [ ] Target block coordinates $(\Delta x, \Delta y, \Delta z)$.
  * [ ] Agent breadcrumb trail (visited positions) and optional BFS optimal reference path.
  * [ ] Live step metrics: step reward, cumulative episode return, remaining distance, Critic estimate $V(s)$, and action probabilities.
  * [ ] Checkpoint metadata: training step count and historical policy/value loss values from the loaded checkpoint.
* [ ] **Single-Inference Runner Script:** Create a CLI script that loads a checkpoint, runs a single pathfinding episode at a configurable playback speed (with pause and step-forward controls), and streams the live LOD + telemetry blobs to the visualizer.
* [ ] **Visualizer Pathfinding Panel:**
  * [ ] **3D Target Highlight:** Render a distinct glowing wireframe/beacon at the target block's coordinates.
  * [ ] **3D Trajectory Line:** Draw a polyline tracing the agent's path from spawn (with a toggle to compare against the optimal BFS path).
  * [ ] **Live Metrics & Loss HUD:** Add a panel displaying step count, distance-to-target, step reward, Critic $V(s)$, action probabilities, and the checkpoint's training/value loss curve.