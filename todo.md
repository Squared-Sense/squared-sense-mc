### Phase 1: Coordinate Frame & Spatial Geometry

*Fix the spatial anchoring so observations are player-centric, rotation-aware, and directional before touching data encoding.*

1. **Convert to Relative Coordinates $(\Delta x, \Delta y, \Delta z)$:**
* Replace absolute world coordinates (`x: 24, y: 61, z: -317`) with real-world block distance relative to Wthe player's feet $(\text{blockPos} - \text{playerPos})$.


* Keep coordinates in actual 1-meter block units across all LODs so matching physical locations share the same values.


2. **Implement Asymmetric Forward/Lateral Offset:**
* For grid size $16$, align the sample box along the player's forward axis such that the player sits at index $2$ (giving $2$ blocks behind and $14$ in front).
* Center the lateral axis symmetrically (indices $[-8, +7]$ relative to the player).


3. **Cardinal Yaw Snapping with Hysteresis:**
* Rotate the sampling axes into egocentric space (so local $+Z$ is always the snapped "forward").
* Snap to the nearest of the 4 cardinal directions ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
* Implement a $\pm 10^\circ$ deadband around the $45^\circ$ diagonal boundaries (snap only changes when rotating past $55^\circ$ or $35^\circ$) to stop grid-thrashing.


4. **Vertical Pitch Translation:**
* Shift the vertical sampling window $Y_{\text{min}} \dots Y_{\text{max}}$ up or down based on the player’s head pitch.
* Looking down translates the box downward (e.g., foot level sits at index $12$, exposing $12$ blocks below); looking up translates it upward.



---

### Phase 2: LOD Scaling & Bounding Boxes

*Lock in the agreed $16\times 16\times 16$ volume and power-of-2 scaling.*

5. **Hardcode $16\times 16\times 16$ Grid Dimensions:**
* Transition the grid dimensions from variable `size: 3` to fixed $16\times 16\times 16$ tensors.




6. **Implement $1\times, 2\times, 4\times$ Stride Sampling:**
* **$B_0$ (LOD 0):** Stride $1$ (spans $16\times 16\times 16$ blocks).
* **$B_1$ (LOD 1):** Stride $2$ (each cell inspects a $2\times 2\times 2$ sub-volume, spanning $32\times 32\times 32$ blocks).
* **$B_2$ (LOD 2):** Stride $4$ (each cell inspects a $4\times 4\times 4$ sub-volume, spanning $64\times 64\times 64$ blocks).



---

### Phase 3: Intelligent Downsampling & Physical Channels

*Move away from naive majority vote and extract physical properties for the 3D CNN.*

7. **Priority Downsampling for Block IDs:**
* Replace naive frequency mode. If a $2\times 2\times 2$ or $4\times 4\times 4$ volume contains high-priority blocks, preserve them in the categorical ID channel even if they are in the minority.
* Priority hierarchy: **Hazards** (Lava, Fire) > **Interactive / Critical** (Chests, Spawners, Ores) > **Liquids** (Water) > **Foliage / Permeable** > **Common Solid** (Stone, Dirt) > **Air**.


8. **Extract Multi-Channel Physical Properties per Voxel:**
* Instead of just `{name, x, y, z}`, compute aggregated feature floats for each cell:


* **Occupancy / Density:** $\frac{\text{solid\_blocks}}{\text{total\_subblocks}} \in [0.0, 1.0]$.
* **Passability:** $1.0$ (air/flowers), $0.5$ (water/cobwebs), $0.0$ (solid).
* **Hazard Flag:** $1.0$ if *any* sub-block contains lava, fire, or cactus; else $0.0$.
* **Hardness / Break Time:** Average or min block hardness in the cell.
* **Light Level:** Combined block + sky light normalized to $[0.0, 1.0]$.
* **Fluid Flow Vector:** Average $(f_x, f_z)$ if liquid is present.




9. **Integer Palette Mapping:**
* Replace string IDs (`minecraft:packed_ice`) with integer indices ($0\dots N$).


* Explicitly reserve ID `0` for `AIR` and ID `1` for `MASKED / UNSEEN`.



---

### Phase 4: Entity Observations ($E_0, E_1, E_2$) & Crosshair Targeting

*Add dynamic agents and high-precision targeting fields.*

10. **Populate Matching $E_n$ Entity Boxes:**
* Query `world.getEntitiesByClass()` within the exact bounding boxes of $B_0, B_1, B_2$.
* Map entity positions into the matching cell indices $(i, j, k)$ in $E_0, E_1, E_2$.
* Channels per entity cell: `entity_count`, `is_hostile` (max-pooled), and `total_health`.


11. **Facing Entity Vector (Crosshair Raycast):**
* Query Minecraft’s native crosshair entity target (`client.targetedEntity` or server-side raycast up to 5 blocks).
* Extract a dedicated feature vector: `[exists (0/1), type_id, delta_x, delta_y, delta_z, vx, vy, vz, health]`.



---

### Phase 5: Visibility & Occlusion Masking

*Remove X-ray omniscience from the block data.*

12. **Exposed-Face Surface Shell Filter (Cull Solid Underworld):**
* For each non-air block queried, check if any of its 6 direct neighbors is air or transparent.
* If a block is completely surrounded by opaque blocks, set its visibility channel to $0.0$ and assign it the `MASKED` token (`special:masked`, ID 1) so the agent does not observe underground caves through solid stone.
* Fully toggleable via in-game command (`/observe visibility [on|off|toggle]`), MMF header flag `FLAG_VISIBILITY_FILTER`, and 3D visualizer UI toggle / Occlusion Heatmap Mode.



---

#### Phase 6: Continuous Tick Hook & Shared Memory Producer

*Moving from the manual `/observe` command to an automatic background stream.*

1. **Tick Hook Cadence:**
* Hook into `ServerTickEvents.END_SERVER_TICK` (or client tick equivalent).
* Run the extractor every $N$ ticks (e.g., $N=1$ for full 20 Hz sync, or $N=2$ to $4$ to conserve CPU cycles).


2. **Double-Buffered State Capture:**
* Read the Minecraft world state on the main game thread during the tick hook.
* Hand the raw block states off to an asynchronous worker thread that handles the LOD aggregation, coordinate translation, and channel encoding. This guarantees zero frame drops or lag spikes in the game loop.


3. **Memory-Mapped File Writer:**
* Initialize a mapped byte buffer pointing to `/dev/shm/mc_observation.bin` (or a OS-specific memory-backed temp file).
* Write serialized tensors directly into RAM using a lightweight spinlock or sequence-number protocol (so consumers never read a half-written frame).



---

#### Phase 7: The Real-Time 3D Visualizer (Debug Monitor)

*A standalone graphical viewport for live inspection and academic figures.*

1. **Visualizer Tech Stack Selection:**
* **Option A (Python / ModernGL or VisPy):** Reads the shared memory buffer directly into NumPy arrays; renders high-speed instanced voxel cubes using OpenGL shaders. Easiest to embed directly into your ML training codebase.
* **Option B (Web-Based / Three.js + WebSocket/Local Bridge):** A lightweight local web app displaying an interactive OrbitControls canvas with a GUI panel. Extremely fast for building interactive slider controls, channel selectors, and toggle switches.


2. **Core Visualizer Capabilities:**
* **LOD Isolation & Blending:** Hotkeys (`1`, `2`, `3`) to view $B_0$, $B_1$, or $B_2$ individually, or render them nested inside one another with transparent boundary cages showing their exact $1\times, 2\times, 4\times$ spatial coverage.
* **Channel Inspection Heatmaps:** A dropdown or shortcut to change voxel coloring:
* *Palette Mode:* True Minecraft block colors.
* *Passability Mode:* Green for walkable, blue for water, red for solid obstacles.
* *Hazard Mode:* Bright glowing red for lava/fire/cactus, black for safe.
* *Light Mode:* Grayscale gradient representing ambient and sky light.
* *Occlusion Mode:* Shading visible surface blocks vs. masked/unseen voxels.


* **Entity Radar Overlay:** Render wireframe red bounding boxes for cells in $E_n$ containing entities, with an arrow indicating the Facing Entity raycast line.
* **Orientation Wireframe:** A visual cone projecting from the player's position indicating actual camera pitch/yaw versus the snapped cardinal grid alignment.
* **Axial Slicing (Voxel Slicer):** Sliders along the $X, Y, Z$ planes to slice open the 3D volume and look inside tunnels, caves, and enclosed structures in real time.



---

#### Phase 8: Model Bridge & Multi-Task Benchmark Hook

*Swapping the consumer from the visualizer to the learning agent.*

1. **PyTorch Shared-Memory Dataset/Env Wrapper:**
* Wrap the shared memory buffer in a standard Gym/Gymnasium environment in Python.
* Convert the mapped byte arrays into zero-copy PyTorch tensors via `torch.frombuffer()`.


2. **Simultaneous Debug Mode:**
* Run the RL training loop on the primary GPU while the visualizer runs concurrently on your secondary monitor, sampling every $k$-th frame from shared memory so you can watch what the agent's spatial senses "see" in real time as policy updates occur.