# Real-Time 3D Observation Visualizer

The **Squared Sense MC 3D Visualizer** is a real-time, zero-overhead diagnostic and debugging tool. It renders the reinforcement learning agent's egocentric multi-LOD spatial observation tensor directly in an interactive WebGL 3D viewport.

Designed to run smoothly on a secondary monitor alongside the Minecraft client, the visualizer reads live frames directly from the shared-memory (MMF) IPC bus at up to 50 Hz, streaming updates to a WebGL frontend powered by Three.js Hardware Instancing.

---

## Architecture Overview

```
                                 [Minecraft Client 1.21.1]
                                             │
                        (Every 2 ticks / 10 Hz ClientTickEvents)
                                             │
                                             ▼
                             [BinaryObservationWriter (Java)]
                                             │
                          (Zero-copy direct off-heap writes)
                                             │
                                             ▼
                        [Shared Memory: squared_sense_mc.bin]
                        ┌───────────────────────────────────┐
                        │ 128-Byte Sequence Header          │
                        │ Slot 0: B0..B2 (485 KB)           │
                        │ Slot 1: B0..B2 (485 KB)           │
                        └───────────────────────────────────┘
                                             │
                                             ▼
                          [visualizer_server.py (Python)]
                                ├── HTTP Server  :8765 (Static Web App & Texture Atlas)
                                └── WebSocket WS :8766 (Live JSON Frames)
                                             │
                                             ▼
                         [Three.js WebGL Frontend (Browser)]
                         ┌─────────────────────────────────┐
                         │ Hardware Instanced Meshes       │
                         │ 512x512 Official Texture Atlas  │
                         │ Multi-LOD Hierarchical Overlap  │
                         │ Physical Channel Heatmaps       │
                         │ Entity Wireframe Bounding Boxes │
                         │ Sightline Laser & Slice Sliders │
                         └─────────────────────────────────┘
```

---

## Key Features

### 1. Hardware-Instanced Official Minecraft Textures (Option B)

Rather than rendering untextured colored cubes, the visualizer extracts authentic 16x16 PNG textures directly from the official Minecraft client JAR:
- **Atlas Compiler (`texture_extractor.py`)**: Locates the Fabric Loom client JAR (`minecraft-clientonly-1.21.1...jar`), extracts 1,012 official block PNGs, and stitches them into a single $512\times 512$ texture atlas (`texture_atlas.png`, 32 tiles per row).
- **Exact UV Mapping (`atlas_uvs.json`)**: Synchronizes with `observations/block_vocab.json`, mapping each integer token ID $0 \dots 1013$ to its exact $(u_0, v_0)$ coordinates on the atlas.
- **Three.js Instanced Attribute**: Injects a custom `instanceUVOffset` attribute into the `MeshLambertMaterial` vertex shader, allowing thousands of cubes to render in a **single WebGL draw call** with authentic pixel art sharpness (`THREE.NearestFilter`).

### 2. Multi-LOD Spatial Hierarchy & Overlap Mode

The visualizer supports both isolated LOD inspection and multi-scale hierarchical overlap:
- **LOD 0 ($B_0$)**: $1\times 1\times 1\text{m}$ cubes centered at $(\Delta x, \Delta y, \Delta z)$ in 1-meter local space.
- **LOD 1 ($B_1$)**: $2\times 2\times 2\text{m}$ cubes representing downsampled $2\times 2\times 2$ world volumes.
- **LOD 2 ($B_2$)**: $4\times 4\times 4\text{m}$ cubes representing $4\times 4\times 4$ regional chunks.
- **Multi-LOD Overlap View**:
  - $B_0$ rendered as solid cubes.
  - $B_1$ rendered as semi-transparent ($35\%$ opacity) larger cubes.
  - $B_2$ rendered as wireframe bounds.
  - Visually demonstrates how the high-resolution immediate action horizon nests inside the coarse macro context.

### 3. Physical Channel Heatmap Overlays

In addition to official block textures, the visualizer can instantly toggle between the model's raw physical observation channels:
- **Hazard Heatmap**: Visualizes environmental dangers (`Hazard` channel). Lava and fire glow bright warning yellow and intense red; safe blocks fade to a cool dark slate.
- **Passability Heatmap**: Visualizes traversal clearance (`Passability` channel). Fully passable air and climbable ladders render in bright emerald green; partial blocks in amber; impassable solid obstacles in dark red.
- **Light Level Heatmap**: Direct representation of Minecraft block/sky illumination ($0.0 \dots 1.0$), transitioning from pitch midnight black to warm amber sunlight.
- **Hardness Heatmap**: Visualizes mining resistance ($0.0 \dots 50.0$), distinguishing soft soil, structural stone, and unbreakable bedrock.

### 4. Entity Spatial Overlays & 3D Mob Preview Heads ($E_0, E_1, E_2$)

Cells containing entities are rendered with both bounding volumes and authentic mob face previews:
- **Wireframe Bounding Boxes**:
  - **Passive / Neutral Mobs**: Bright green wireframe box (`#39ff14`).
  - **Hostile Mobs**: Pulsing red wireframe box (`#ff3366`) animated at 6 Hz.
- **3D Mob Preview Heads**:
  - Centered inside the cell, an instanced 3D head cube ($0.55\text{m} \times 0.55\text{m} \times 0.55\text{m}$) renders the authentic face texture extracted directly from the Minecraft JAR (e.g. Zombie, Skeleton, Creeper, Spider, Enderman, Cow, Pig, Sheep, Villager, Steve).
  - Built from [`entity_heads.png`](file:///c:/Users/Krishna/Projects/Squared%20Sence%20MC/visualizer/backend/assets/entity_heads.png) and [`entity_heads.json`](file:///c:/Users/Krishna/Projects/Squared%20Sence%20MC/visualizer/backend/assets/entity_heads.json).
  - **Camera-Oriented Facing**: The head cube rotates smoothly around its vertical axis to face the camera, ensuring the mob face is clearly recognizable from any viewing angle.
  - Entity count, hostility status, and health pool are inspected dynamically on hover.

### 5. Player Orientation, Anchor & Sightline Laser

- **Player Foot Anchor**: Bounding box ($0.6\text{m} \times 1.8\text{m} \times 0.6\text{m}$) anchored at $(0, 0, 0)$ with a horizontal forward-facing arrow.
- **Head Sightline Laser**: Dynamic laser ray projecting 24 meters from the player's eye level $(0, 1.62, 0)$ along the exact camera $(\text{yaw}, \text{pitch})$ vector, tipped with a glowing target reticle.
- **Ground Reference Grid**: $32\times 32\text{m}$ coordinate grid on the floor.

### 6. Interactive Slice Planes (Cross-Section Clipping)

Six independent range sliders (Lateral X, Vertical Y, Forward Z) enable clipping the 3D volume down to any slice:
- Isolate underground caves or vertical cross-sections in front of the player.
- Strip away ceilings to inspect floor hazards beneath overhangs.

### 7. Voxel Raycast Inspection (Tooltip HUD)

Hovering over any voxel in the 3D viewport casts a hardware raycast, displaying an on-screen telemetry card:
- Block registry name and token ID.
- Grid indices $(gx, gy, gz)$ and egocentric metric offsets $(\Delta x, \Delta y, \Delta z)$.
- Channel values: Occupancy, Passability, Hazard, Hardness, Light Level.
- Entity information: count, hostility status, and combined health pool.

---

## Directory Structure

All visualizer source code and static assets are located in `visualizer/`:

```
visualizer/
├── backend/
│   ├── assets/
│   │   ├── atlas_uvs.json        # 1,014 mapped token UV coordinates
│   │   ├── texture_atlas.png     # 512x512 official 16x16 PNG texture atlas
│   │   ├── entity_heads.json     # 19 mapped mob preview head UV coordinates
│   │   └── entity_heads.png      # 128x128 official mob face texture atlas
│   ├── head_extractor.py         # JAR mob face extractor & head atlas builder
│   ├── texture_extractor.py      # JAR block extractor & texture atlas stitcher
│   └── visualizer_server.py      # MMF IPC reader, HTTP & WebSocket server
└── frontend/
    ├── css/
    │   └── style.css             # Dark cyberpunk telemetry theme
    ├── js/
    │   └── app.js                # Three.js instanced rendering & controls
    ├── vendor/
    │   ├── OrbitControls.js      # Three.js camera interaction
    │   └── three.min.js          # Three.js r128 WebGL library
    └── index.html                # HUD layout and control panels
```

---

## How to Run the Visualizer

### Step 1: Start the Visualizer Backend

In a terminal, run:

```powershell
python visualizer/backend/visualizer_server.py
```

Optional flags:
- `--port 8765`: Custom HTTP port for frontend (default: 8765).
- `--ws-port 8766`: Custom WebSocket port for live streaming (default: 8766).
- `--demo`: Generates a synthetic Minecraft observation scene (terrain, tree, water, lava, zombie, cow) for verification without running Minecraft.

### Step 2: Open the Web App

Open any modern browser (Chrome, Edge, Firefox) and navigate to:

```
http://localhost:8765
```

The status pill in the top header will transition to **LIVE STREAM** (green).

### Step 3: Stream from Minecraft

In the Minecraft client, start live streaming at the desired tick interval:

```
/observe stream start 2
```

*(Or alias `/obs stream start 2`). Default cadence is every 2 ticks (10 Hz).*

As the player moves, turns, or digs, the 3D viewport on your second monitor will mirror the egocentric observation slice in real time.

---

## Keyboard & Mouse Navigation

| Control | Action |
| :--- | :--- |
| **Left Click + Drag** | Orbit camera around target |
| **Right Click + Drag** | Pan camera |
| **Scroll Wheel** | Zoom in / out |
| **Hover Mouse** | Inspect voxel properties and entity status |
| **Camera View Buttons** | Switch between Isometric, Top-Down, Side, and First-Person views |
| **LOD Buttons** | Toggle between LOD 0 (1m), LOD 1 (2m), LOD 2 (4m), and Multi-LOD Overlap |
| **Render Mode Buttons** | Switch between Official Textures, Hazard, Passability, Light, and Hardness |

