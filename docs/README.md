# Squared Sense MC — Documentation

Welcome to the internal documentation for **Squared Sense MC**, a client-side Minecraft Fabric mod designed for Reinforcement Learning (RL) observation extraction and spatial perception.

## Documentation Index

1. **[Architecture Overview](ARCHITECTURE.md)**
   - High-level design and RL observation pipeline.
   - Package layout and component responsibilities.
   - Performance optimizations for fixed-size tensor sampling.

2. **[Spatial Geometry & Coordinate Frame](SPATIAL_GEOMETRY.md)**
   - Relative coordinates $(\Delta x, \Delta y, \Delta z)$ in 1-meter block units.
   - Egocentric coordinate frame ($X_{\text{lateral}}, Y_{\text{vertical}}, Z_{\text{forward}}$).
   - Asymmetric forward/lateral offsets (player at $gz = 2, gx = 8$).
   - Cardinal yaw snapping with $\pm 10^\circ$ hysteresis deadband.
   - Vertical pitch translation (foot level dynamically shifting between indices 1, 6, and 12).

3. **[Block Priority & Classification System](BLOCK_PRIORITY_SYSTEM.md)**
   - The 6-rank priority hierarchy: Hazards > Interactive/Critical > Liquids > Foliage > Solid > Air.
   - Intelligent downsampling algorithm ensuring rare/critical blocks are not erased.
   - Classification rules, singleton pointer caching, and case studies.

4. **[Multi-Channel Physical Feature Extraction](PHYSICAL_CHANNELS.md)**
   - The 6 physical property channels per voxel: Occupancy, Passability, Hazard Flag, Hardness, Light Level, and Fluid Flow.
   - Egocentric rotation of fluid velocity vectors.
   - Sub-volume aggregation rules and PyTorch tensor conversion.

5. **[Integer Palette Mapping & Vocabulary](INTEGER_PALETTE_MAPPING.md)**
   - Static canonical integer tokens ($0 \dots N$) frozen alphabetically.
   - Reserved tokens: ID 0 = `minecraft:air`, ID 1 = `special:masked`.
   - Dynamic caching, auto-export to `block_vocab.json`, and `/observe vocab` command.
   - PyTorch `nn.Embedding` integration with 1:1 synchronization.

6. **[Entity Observations & Grids](ENTITY_OBSERVATIONS.md)**
   - Matching $16\times 16\times 16$ entity tensor grids ($E_0, E_1, E_2$) matching block grids $B_0, B_1, B_2$.
   - Three channels per cell: `entityCount`, `isHostile` (max-pooled), and `totalHealth`.
   - Hostility classification (`Monster`, active `Angerable`) and health extraction.
   - Dual-stream 3D CNN PyTorch integration guide.

7. **[Shared-Memory IPC & Streaming](SHARED_MEMORY_IPC.md)**
   - High-throughput Memory-Mapped File (MMF) IPC bus (`/dev/shm` on Linux, `%TEMP%` on Windows).
   - Lock-free Ping-Pong double-buffering for sub-millisecond zero-tear reads.
   - Exact binary layout for $B_0..B_2$ and $E_0..E_2$ tensors.
   - Zero-copy Python NumPy and PyTorch streaming.

8. **[Observation Specification & Commands](OBSERVATION_SPEC.md)**
   - Fixed $16\times 16\times 16$ grid tensors across LODs 0, 1, and 2 (strides 1, 2, and 4).
   - Priority downsampling integration and tie-breaking algorithms.
   - In-game commands (`/observe`, `/obs`, `/squish`).
   - JSON output schema and Python consumption examples.

