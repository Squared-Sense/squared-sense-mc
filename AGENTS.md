# AGENTS.md — Intent-Based Navigation & System Index

> **Notice for AI Agents**: This file serves as the primary router for AI agents working in the **Squared Sense MC** repository. Instead of reading all documents into your context window and causing token bloat, consult the **[Agent Intent Routing Matrix](#agent-intent-routing-matrix)** below to pinpoint the exact specification and source files needed for your specific task.

> Also check [README.md](/DOCS/README.md)

---

## Critical System Invariants & Ground Rules

All AI agents operating in this workspace must strictly follow these rules:

1. **Manual Build & Git Management**:
   * **NEVER execute Gradle or Git commands** (`./gradlew build`, `./gradlew runClient`, `git commit`, `git push`, etc.).
   * The user builds, launches the Minecraft client, and manages version control manually.
2. **Indentation Standards**:
   * Maintain **strict 4-space indentation** across all files: Java (`.java`), Python (`.py`), JavaScript (`.js`), HTML (`.html`), CSS (`.css`), and Markdown (`.md`).
3. **Zero-Allocation Hot Path**:
   * Continuous observation sampling executes at 10–20 Hz on the client tick loop.
   * **Never instantiate objects (`new`) inside inner sampling loops** (e.g., inside `BlockSampler`, `BinaryObservationWriter`, `PhysicalPropertyExtractor`).
   * Reuse pre-allocated `BlockPos.Mutable`, identity maps, and stack primitives to avoid triggering JVM garbage collection stutters.
4. **File Link Standards**:
   * All referenced files must be clickable markdown links relative to the project root directory (`Squared Sence MC`) (e.g., `[ARCHITECTURE.md](/docs/ARCHITECTURE.md)`).

---

## Agent Intent Routing Matrix

Find your task intent in the table below and read **only** the designated primary documentation and code files:

| Agent Intent / Goal | Primary Documentation | Key Source Files | Core Concept / Invariant |
|---|---|---|---|
| **Understand overall system architecture & pipeline** | [docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md) | [SquaredSenseMCClient.java](/src/client/java/org/sqsense/squared_sense_mc/client/SquaredSenseMCClient.java)<br>[ObservationConfig.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ObservationConfig.java) | Client-side Fabric mod extracting $16\times 16\times 16$ 3D voxel tensors for RL perception. |
| **Work on spatial geometry, camera, or coordinates** | [docs/SPATIAL_GEOMETRY.md](/docs/SPATIAL_GEOMETRY.md) | [SnappedCardinal.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/SnappedCardinal.java)<br>[YawHysteresisTracker.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/YawHysteresisTracker.java)<br>[PitchWindowCalculator.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/PitchWindowCalculator.java)<br>[ObservationTransform.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/ObservationTransform.java) | Egocentric coordinate frame ($+Z$ forward, $+X$ right); player at $(gx=8, gz=2)$; $\pm 10^\circ$ hysteresis deadband; pitch-dependent vertical window. |
| **Work on downsampling & block priority hierarchy** | [docs/BLOCK_PRIORITY_SYSTEM.md](/docs/BLOCK_PRIORITY_SYSTEM.md) | [BlockPriority.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/priority/BlockPriority.java)<br>[BlockPriorityRegistry.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/priority/BlockPriorityRegistry.java)<br>[BlockSampler.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/sampler/BlockSampler.java) | 6-tier hierarchy: HAZARD > INTERACTIVE > LIQUID > FOLIAGE > SOLID > AIR. Critical rare blocks survive LOD 1/2 downsampling. |
| **Work on physical voxel channels & lighting** | [docs/PHYSICAL_CHANNELS.md](/docs/PHYSICAL_CHANNELS.md) | [PhysicalPropertyExtractor.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/channels/PhysicalPropertyExtractor.java)<br>[VoxelPhysicalProperties.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/channels/VoxelPhysicalProperties.java) | 7 physical channels: Occupancy, Passability, Hazard, Hardness, Light, FlowX, FlowZ. Solid block light is surface-resolved incident light. |
| **Work on integer palette & block tokens** | [docs/INTEGER_PALETTE_MAPPING.md](/docs/INTEGER_PALETTE_MAPPING.md) | [BlockVocabulary.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/palette/BlockVocabulary.java)<br>[block_vocab.json](/observations/block_vocab.json) | Static alphabetical vocabulary ($0 \dots N$); ID 0 = `minecraft:air`, ID 1 = `special:masked`; synchronized with PyTorch `nn.Embedding`. |
| **Work on entity observations & hostility tracking** | [docs/ENTITY_OBSERVATIONS.md](/docs/ENTITY_OBSERVATIONS.md) | [EntityClassifier.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/entity/EntityClassifier.java)<br>[EntityGridSampler.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/entity/EntityGridSampler.java)<br>[EntityEntry.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/entity/EntityEntry.java) | Matching $16\times 16\times 16$ entity grids ($E_0, E_1, E_2$); channels: count, max-pooled hostility, total health; Monster vs passive mob detection. |
| **Work on visibility & surface shell culling (Phase 5)** | [docs/BLOCK_PRIORITY_SYSTEM.md](/docs/BLOCK_PRIORITY_SYSTEM.md)<br>[docs/ARCHITECTURE.md](/docs/ARCHITECTURE.md) | [SurfaceShellFilter.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/visibility/SurfaceShellFilter.java)<br>[BinaryObservationWriter.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/BinaryObservationWriter.java) | Culls buried underworld blocks completely enclosed by solid cubes to ID 1 (`special:masked`) with `visibility = 0.0`. Toggleable via `/observe visibility`. |
| **Work on shared memory IPC & binary layout** | [docs/SHARED_MEMORY_IPC.md](/docs/SHARED_MEMORY_IPC.md) | [IpcBufferLayout.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/IpcBufferLayout.java)<br>[SharedMemoryChannel.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/SharedMemoryChannel.java)<br>[BinaryObservationWriter.java](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/BinaryObservationWriter.java) | Protocol v3, lock-free Ping-Pong double buffering, 1.04 MB shared buffer (`squared_sense_mc.bin`). Sub-millisecond zero-tear updates. |
| **Consume observations in Python / RL model bridge** | [docs/SHARED_MEMORY_IPC.md#python-consumption](/docs/SHARED_MEMORY_IPC.md) | [mc_ipc_layout.py](/bridge/mc_ipc_layout.py)<br>[mc_ipc_reader.py](/bridge/mc_ipc_reader.py) | Python zero-copy reader using `mmap` and `numpy.frombuffer()`. Returns structured dicts of NumPy tensors for Gym/PyTorch. |
| **Interface with World Model & Training Repo** | [docs/WORLD_MODEL_CONTRACTS.md](/docs/WORLD_MODEL_CONTRACTS.md) | [docs/WORLD_MODEL_CONTRACTS.md](/docs/WORLD_MODEL_CONTRACTS.md) | 16-channel continuous tensor contract, channel-last layout, 12-action space, JAX/Flax NNX compatibility in companion repo. |
| **Work on offline RL training & parallel workers** | [docs/WORLD_MODEL_CONTRACTS.md](/docs/WORLD_MODEL_CONTRACTS.md)<br>[plan.md](/plan.md) | `../Squared-Sense-World-Model/eval/train_ppo_nav.py`<br>`../Squared-Sense-World-Model/env/` | Relocated to companion repo: Real MinecraftNavEnv, SpatialJEPA + ActorCriticHead, GAE, PPO loss, BC warm-start. |
| **Work on single inference & visualizer telemetry** | [docs/VISUALIZER.md](/docs/VISUALIZER.md)<br>[plan.md](/plan.md) | `../Squared-Sense-World-Model/eval/run_inference.py`<br>[visualizer/backend/visualizer_server.py](/visualizer/backend/visualizer_server.py) | Single-episode playback runner streaming live MMF frames + TELM v1 inference telemetry directly into the 3D WebGL visualizer. |
| **Work on 3D real-time visualizer** | [docs/VISUALIZER.md](/docs/VISUALIZER.md) | [visualizer_server.py](/visualizer/backend/visualizer_server.py)<br>[app.js](/visualizer/frontend/js/app.js)<br>[index.html](/visualizer/frontend/index.html) | Standalone WebGL viewer (Three.js instanced rendering), 512x512 official texture atlas, heatmaps, entity boxes, axial slice planes. |
| **Inspect in-game commands & manual export** | [docs/OBSERVATION_SPEC.md](/docs/OBSERVATION_SPEC.md) | [ObserveCommand.java](/src/client/java/org/sqsense/squared_sense_mc/client/command/ObserveCommand.java)<br>[SquishCommand.java](/src/client/java/org/sqsense/squared_sense_mc/client/command/SquishCommand.java) | Command dispatcher for `/observe [all|0|1|2|stream|visibility|vocab]` and `/squish <size>`. Asynchronous JSON serializer. |
| **Check project roadmap, completed phases & pending tasks** | [todo.md](/todo.md) | [todo.md](/todo.md) | Tracks Phases 1 through 8 from geometry anchoring to RL Gym benchmark hooks. |

---

## Deep-Dive Intent Guides

### 1. Spatial Geometry & Coordinate Framing
* **When to read**: You are adjusting how the sample bounding boxes move, rotate, or anchor around the player.
* **Key Spec**: [docs/SPATIAL_GEOMETRY.md](/docs/SPATIAL_GEOMETRY.md)
* **Core Code**:
  * [`SnappedCardinal.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/SnappedCardinal.java): Snaps raw continuous yaw to 4 cardinal headings (`SOUTH`, `WEST`, `NORTH`, `EAST`) and provides forward/right basis vectors.
  * [`YawHysteresisTracker.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/YawHysteresisTracker.java): Implements a $\pm 10^\circ$ deadband window around $45^\circ$ diagonals to prevent rapid grid-thrashing when moving diagonally.
  * [`PitchWindowCalculator.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/PitchWindowCalculator.java): Translates player pitch $[-90^\circ, +90^\circ]$ to a vertical foot index:
    * Looking straight ahead: foot index $gy = 6$ (6 blocks below, 9 above).
    * Looking straight down: foot index $gy = 12$ (12 blocks below, 3 above).
    * Looking straight up: foot index $gy = 1$ (1 block below, 14 above).
  * [`ObservationTransform.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/geometry/ObservationTransform.java): Asymmetrically anchors player at grid index $(gx=8, gz=2)$, placing 2 blocks behind and 14 ahead.

### 2. Block Priority & Intelligent Downsampling
* **When to read**: You are modifying multi-meter LOD aggregation ($B_1$ stride 2, $B_2$ stride 4) or ensuring rare blocks survive downsampling.
* **Key Spec**: [docs/BLOCK_PRIORITY_SYSTEM.md](/docs/BLOCK_PRIORITY_SYSTEM.md)
* **Core Code**:
  * [`BlockPriority.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/priority/BlockPriority.java): Enumerates the 6 priority tiers: `HAZARD > INTERACTIVE > LIQUID > FOLIAGE > SOLID > AIR`.
  * [`BlockPriorityRegistry.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/priority/BlockPriorityRegistry.java): Provides cached $O(1)$ block-to-priority classification using an `IdentityHashMap` backed by vanilla block tags.
  * [`BlockSampler.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/sampler/BlockSampler.java): Traverses sub-volumes and picks the most frequent block *within the highest non-air priority tier present*.

### 3. Physical Channels & Lighting Engine
* **When to read**: You are extracting physical properties (passability, hardness, hazards, fluid flow, light level).
* **Key Spec**: [docs/PHYSICAL_CHANNELS.md](/docs/PHYSICAL_CHANNELS.md)
* **Core Code**:
  * [`PhysicalPropertyExtractor.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/channels/PhysicalPropertyExtractor.java): Computes continuous physical features per cell and aggregates sub-volumes.
  * **Vanilla Light Level Invariant**:
    * Full solid cubes (Stone, Dirt, Wood) have opacity 15, so querying `world.getLightLevel(pos)` directly inside them returns 0.
    * `PhysicalPropertyExtractor.getNormalizedLight(...)` resolves solid block illumination from the **maximum incident light among its exposed adjacent open faces** (`world.getLightLevel(neighborPos)`), exactly mirroring vanilla `BlockModelRenderer`.
    * Fluid flow vectors $(v_x, v_z)$ are dynamically rotated into egocentric space via `cardinal.getRightX()`, `cardinal.getForwardZ()`.

### 4. Visibility & Occlusion Masking (Phase 5)
* **When to read**: You are working on subterranean culling, exposed-shell filtering, or x-ray prevention.
* **Core Code**:
  * [`SurfaceShellFilter.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/visibility/SurfaceShellFilter.java): Tests the 6 cardinal faces of blocks (and outer bounding faces of sub-volumes). If completely enclosed by opaque full cubes, marks the block as occluded.
  * In `BinaryObservationWriter` and `BlockSampler`: occluded solid blocks are assigned `blockId = BlockVocabulary.MASKED_ID` (ID 1) with `visibility = 0.0f`.
  * In-game toggle: `/observe visibility [on|off|status]`.

### 5. Shared Memory (MMF) IPC & Real-Time Streaming
* **When to read**: You are debugging or extending shared-memory IPC, Ping-Pong buffering, or protocol versions.
* **Key Spec**: [docs/SHARED_MEMORY_IPC.md](/docs/SHARED_MEMORY_IPC.md)
* **Core Code**:
  * [`IpcBufferLayout.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/IpcBufferLayout.java): Protocol version 3 layout. File size: 1,093,632 bytes (~1.04 MB). Double-buffered slots at offsets `HEADER_BYTES` (128) and `HEADER_BYTES + SLOT_SIZE` (544,896).
  * [`SharedMemoryChannel.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/SharedMemoryChannel.java): OS-independent path (`/dev/shm` on Linux, `%TEMP%` on Windows). Memory-fenced double buffering.
  * [`BinaryObservationWriter.java`](/src/client/java/org/sqsense/squared_sense_mc/client/observation/ipc/BinaryObservationWriter.java): Writes block and entity tensors directly into `MappedByteBuffer` with zero allocations.
  * [`mc_ipc_reader.py`](/bridge/mc_ipc_reader.py): Python reader that maps the file with `mmap` and reconstructs NumPy tensors using zero-copy slicing.

### 6. Real-Time 3D WebGL Visualizer
* **When to read**: You are modifying the browser visualizer, adding overlays, adjusting HUD controls, or tweaking shaders.
* **Key Spec**: [docs/VISUALIZER.md](/docs/VISUALIZER.md)
* **Core Code**:
  * [`visualizer_server.py`](/visualizer/backend/visualizer_server.py): Reads live MMF frames, serves static web assets on HTTP 8765, and pushes JSON frames over WebSocket 8766.
  * [`app.js`](/visualizer/frontend/js/app.js): Three.js frontend featuring hardware-instanced rendering (`THREE.InstancedMesh`), official texture atlas UVs, entity heads overlay, light/hazard/passability heatmaps, and axial clipping planes.
  * [`index.html`](/visualizer/frontend/index.html): Responsive HUD with collapsible panels for split-screen monitoring alongside Minecraft.

---

## Directory & Package Quick Reference

```
Squared Sence MC/
├── AGENTS.md                               # <-- You are here (primary AI agent index)
├── README.md                               # Project overview and quickstart
├── todo.md                                 # Phase 1-8 development roadmap and progress
├── docs/                                   # Detailed system specifications
│   ├── ARCHITECTURE.md                     # High-level architecture and component layout
│   ├── SPATIAL_GEOMETRY.md                 # Coordinate frame, yaw snapping, pitch translation
│   ├── BLOCK_PRIORITY_SYSTEM.md            # Priority hierarchy and downsampling rules
│   ├── PHYSICAL_CHANNELS.md                # Continuous physical channels and light math
│   ├── INTEGER_PALETTE_MAPPING.md          # Alphabetical canonical vocabulary (ID 0..N)
│   ├── ENTITY_OBSERVATIONS.md              # 3D entity grids, hostility, and health channels
│   ├── SHARED_MEMORY_IPC.md                # Memory-mapped file protocol & double buffering
│   ├── OBSERVATION_SPEC.md                 # Observation schemas and command syntax
│   ├── VISUALIZER.md                       # Real-time WebGL 3D diagnostic visualizer
│   └── WORLD_MODEL_CONTRACTS.md            # Companion World Model 16-channel & JAX contracts
├── bridge/                                 # Python / ML IPC bridge
│   ├── mc_ipc_layout.py                    # Binary IPC layout constants for Python
│   └── mc_ipc_reader.py                    # Zero-copy NumPy/PyTorch MMF reader
├── visualizer/                             # 3D Visualizer application
│   ├── backend/
│   │   ├── visualizer_server.py            # HTTP & WebSocket server reading live MMF
│   │   ├── texture_extractor.py            # Texture atlas stitcher from client JAR
│   │   └── assets/                         # Texture atlas (512x512) and UV mappings
│   └── frontend/
│       ├── index.html                      # WebGL viewer HUD & telemetry panel
│       ├── css/style.css                   # Cyberpunk HUD styling
│       └── js/app.js                       # Three.js hardware instancing and telemetry
└── src/client/java/org/sqsense/squared_sense_mc/client/
    ├── SquaredSenseMCClient.java           # Client mod entrypoint
    ├── command/                            # In-game chat commands (/observe, /squish)
    ├── observation/
    │   ├── ObservationConfig.java          # Global constants and runtime toggles
    │   ├── channels/                       # VoxelPhysicalProperties & extractor
    │   ├── entity/                         # EntityClassifier, EntityGridSampler
    │   ├── geometry/                       # SnappedCardinal, Hysteresis, PitchWindow
    │   ├── ipc/                            # IpcBufferLayout, BinaryObservationWriter, Channel
    │   ├── palette/                        # BlockVocabulary
    │   ├── priority/                       # BlockPriority, BlockPriorityRegistry
    │   ├── sampler/                        # BlockSampler
    │   ├── serialization/                  # JSON serialization for manual dumps
    │   └── visibility/                     # SurfaceShellFilter (Phase 5 culling)
    └── util/                               # Raycast and inspection utilities
```

