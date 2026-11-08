# Architecture Overview

## 1. Project Goal & High-Level Purpose

**Squared Sense MC** is an observational data retrieval system built as a client-side Fabric mod for Minecraft 1.21.1. 

Traditional RL approaches for Minecraft rely on processing RGB video streams, which demands heavy convolutional encoders and struggles with depth ambiguity. Squared Sense MC provides a **direct 3D block perception pipeline**:
- Extracts structured 3D spatial grids centered on the player.
- Operates across multiple Levels of Detail (LODs) to capture both immediate surroundings and macro structures.
- Projects observations into an **egocentric reference frame** where $+Z$ is always the snapped forward direction and $+X$ is lateral right.
- Anchors all block positions to physical 1-meter relative distances from the player's feet.

---

## 2. Package & Component Structure

To prevent bloated, monolithic files, the codebase is partitioned into distinct single-responsibility modules:

```
src/client/java/org/sqsense/squared_sense_mc/client/
├── SquaredSenseMCClient.java              # Client mod entrypoint
├── command/
│   ├── ModClientCommands.java            # Command dispatcher registration callback
│   ├── ObserveCommand.java               # Command router for /observe and /obs
│   ├── SquishCommand.java                # Arbitrary-size debug export command
│   ├── BlockInfoCommand.java             # Single-block inspector (/sense <pos>)
│   └── TargetBlockInfoCommand.java       # Crosshair raycast inspector (/sense_target)
├── observation/
│   ├── ObservationConfig.java            # Central constants (grid size, strides, anchor indices)
│   ├── channels/
│   │   ├── VoxelPhysicalProperties.java  # 6-channel physical property data record
│   │   └── PhysicalPropertyExtractor.java# Density, passability, hazard, hardness, light & fluid flow
│   ├── geometry/
│   │   ├── SnappedCardinal.java          # 4 cardinal directions & orthonormal rotation math
│   │   ├── YawHysteresisTracker.java     # Stateful ±10° deadband yaw stabilization
│   │   ├── PitchWindowCalculator.java    # Dynamic vertical foot index translation from head pitch
│   │   └── ObservationTransform.java     # Unified spatial frame transform (local -> world delta)
│   ├── priority/
│   │   ├── BlockPriority.java            # Priority tier enum (HAZARD > INTERACTIVE > LIQUID > FOLIAGE > SOLID > AIR)
│   │   └── BlockPriorityRegistry.java    # Cached O(1) block-to-priority classifier with tag fallbacks
│   ├── palette/
│   │   └── BlockVocabulary.java          # Canonical alphabetical token ID dictionary & JSON exporter
│   ├── entity/
│   │   ├── EntityEntry.java              # Individual entity details record (type, health, hostile, local pos)
│   │   ├── EntityClassifier.java         # Hostility (Monster, Angerable) & health extractor
│   │   ├── EntitySamplingResult.java     # Record holding 16x16x16 entity tensor and aggregate counts
│   │   └── EntityGridSampler.java        # Entity bounding box query and 3D tensor cell projection
│   ├── sampler/
│   │   ├── BlockSampler.java             # Zero-allocation block sampling and priority downsampling
│   │   └── SamplingResult.java           # Record holding grid tensor array and cell statistics
│   └── serialization/
│       └── ObservationJsonSerializer.java# Metadata generation and async file writing
└── util/
    └── BlockInspector.java               # Block raycast utilities
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| `ObservationConfig` | Global single source of truth for observation dimensions, anchor indices, and LOD strides. |
| `VoxelPhysicalProperties` | 6-channel continuous feature record (occupancy, passability, hazard, hardness, lightLevel, fluidFlow). |
| `PhysicalPropertyExtractor` | High-speed single-block and sub-volume physical feature aggregator with egocentric fluid flow rotation. |
| `SnappedCardinal` | Represents the 4 cardinal directions (SOUTH, WEST, NORTH, EAST) and applies 2D rotation. |
| `YawHysteresisTracker` | Tracks player rotation statefully, preventing oscillation/thrashing around 45° boundaries. |
| `PitchWindowCalculator` | Pure function mapping player pitch $[-90^\circ, +90^\circ]$ to vertical foot index $gy \in [1, 12]$. |
| `ObservationTransform` | Encapsulates camera state and maps tensor indices $(gx, gy, gz) \in [0..15]^3$ to world relative offsets $(\Delta x, \Delta y, \Delta z)$. |
| `BlockPriority` | 6-tier priority hierarchy enum (HAZARD, INTERACTIVE, LIQUID, FOLIAGE, SOLID, AIR). |
| `BlockPriorityRegistry` | High-speed singleton block-to-priority classifier with cached lookups and vanilla tag fallbacks. |
| `BlockVocabulary` | Static canonical vocabulary mapping Minecraft blocks to frozen integer token IDs (0=Air, 1=Masked, 2..N=Blocks) with $O(1)$ lookups and JSON export. |
| `EntityClassifier` | Categorizes entities by hostility (`Monster`, active `Angerable`) and extracts health values. |
| `EntityGridSampler` | Queries client world entities in the matching observation bounding box and projects them into $16\times 16\times 16$ entity tensor grids ($E_0, E_1, E_2$). |
| `BlockSampler` | Fast traversal of the client world using cached block IDs, mutable coordinate objects, and priority downsampling. |
| `ObservationJsonSerializer` | Assembles egocentric JSON payloads and performs non-blocking I/O via `CompletableFuture.runAsync`. |
| `ObserveCommand` | Brigadier command parsing, validation, and user feedback formatting. |

---

## 3. Performance & Optimization Strategies

Extracting up to $64\times 64\times 64 = 262,144$ blocks per observation on the client render thread requires careful memory and CPU budgeting:

1. **Zero-Allocation Coordinate Traversal (`BlockPos.Mutable`)**:
   - Rather than instantiating $4,096$ to $262,144$ `new BlockPos(x, y, z)` objects per capture, sampling uses a single reused `BlockPos.Mutable`. This completely eliminates heap garbage collection pressure during sampling.

2. **Interned Block Identifier Cache**:
   - `Registries.BLOCK.getId(block).toString()` formats a new `String` on every invocation.
   - `BlockSampler` maintains a thread-safe `ConcurrentHashMap<Block, String>` cache. Since `Block` instances are singletons in Minecraft, this lookup runs in $O(1)$ without allocating strings.

3. **Single Reused Frequency Map for Mode Downsampling**:
   - For LOD 1 (stride 2, 8 blocks/cell) and LOD 2 (stride 4, 64 blocks/cell), a single `Map<String, Integer> cellCounts = new HashMap<>(32)` is instantiated once and `.clear()` is called between cells, avoiding 4,096 map allocations per pass.

4. **LOD 0 Direct Fast-Path**:
   - When `stride == 1` (LOD 0), sub-volume iteration and frequency counting are bypassed entirely for direct $1:1$ cell assignment.

5. **Asynchronous Disk Serialization**:
   - World block states are read synchronously on the client thread where chunk data is consistent.
   - Once data is collected in memory, JSON formatting and disk I/O are offloaded to `CompletableFuture.runAsync()`, ensuring game ticks never stutter or drop frames.

