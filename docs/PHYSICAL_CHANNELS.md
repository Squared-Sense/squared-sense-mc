# Multi-Channel Physical Feature Extraction

This document specifies the **6 continuous physical feature channels** extracted per voxel cell, implemented in **Phase 3 (Task 8)**.

---

## 1. Motivation: Beyond Categorical Block Names

While discrete block IDs (e.g. `minecraft:stone`) provide semantic identity, neural networks (specifically 3D Convolutional Neural Networks) perform significantly better when provided with **dense, continuous spatial channels**:
- **Occupancy / Density**: Distinguishes open air, partial foliage, and dense subterranean solid matter.
- **Passability**: Enables the agent to evaluate traversal costs and path obstacles without memorizing collision shapes of hundreds of blocks.
- **Hazard Awareness**: Directly alerts the policy to dangerous environmental lethal zones.
- **Hardness**: Allows navigation and mining policies to assess excavation difficulty and tool wear.
- **Light Level**: Signals hostile mob spawn likelihood and visibility conditions.
- **Fluid Flow**: Enables the agent to anticipate water current forces (pushing forward, sideways, or backward).

---

## 2. Channel Specifications

Each cell in the $16\times 16\times 16$ grid contains the categorical block name, relative coordinates $(\Delta x, \Delta y, \Delta z)$, and the following 6 physical property channels:

| Channel | Data Type | Range | Definition & Aggregation |
|---|---|---|---|
| **`occupancy`** | Float | $[0.0, 1.0]$ | Ratio of solid blocks to total sub-blocks: $\frac{\text{solid\_blocks}}{\text{total\_subblocks}}$. $1.0$ for solid bulk, $0.0$ for air/liquid. |
| **`passability`** | Float | $[0.0, 1.0]$ | Movement permeability: $1.0$ (air/flowers/grass), $0.5$ (water/cobwebs/powder snow/vines), $0.0$ (solid obstacles). Averaged across sub-blocks. |
| **`hazard`** | Float | $\{0.0, 1.0\}$ | **Binary hazard trigger**: $1.0$ if **any** sub-block in the cell contains lava, fire, cactus, magma, etc.; otherwise $0.0$. |
| **`hardness`** | Float | $[0.0, 100.0]$ | Average break hardness across sub-blocks ($0.0$ for air, $0.5$ dirt, $1.5$ stone, $50.0$ obsidian, $100.0$ unbreakable bedrock). |
| **`lightLevel`** | Float | $[0.0, 1.0]$ | Combined block + sky light level normalized: $\frac{\text{lightLevel}}{15.0}$. Averaged across sub-blocks. |
| **`fluidFlow`** | Float[2] | $[-1.0, 1.0]^2$ | Fluid velocity vector $[f_x, f_z]$ **rotated into the egocentric camera frame** ($+f_z$ forward, $+f_x$ right). |

---

## 3. Egocentric Rotation of Fluid Flow Vectors

In Minecraft, fluid velocity is reported in world space $(v_x, v_z)$. If an agent moves facing North, a river flowing South pushes it from the front; if facing South, the same river pushes it from behind.

To make the fluid flow invariant to world orientation, the world velocity is projected onto the snapped cardinal axes:

$$f_x = v_x \cdot R_x + v_z \cdot R_z \quad (\text{Lateral Right})$$
$$f_z = v_x \cdot F_x + v_z \cdot F_z \quad (\text{Forward})$$

Where $\vec{F} = (F_x, F_z)$ is the snapped forward vector and $\vec{R} = (R_x, R_z)$ is the snapped right vector:
- $+f_z > 0$: Fluid pushes the agent **forward**.
- $-f_z < 0$: Fluid pushes the agent **backward**.
- $+f_x > 0$: Fluid pushes the agent to the **right**.
- $-f_x < 0$: Fluid pushes the agent to the **left**.

---

## 4. Sub-Volume Aggregation Rules (LOD 1 & LOD 2)

| Channel | LOD 0 ($1\times 1\times 1$) | LOD 1 ($2\times 2\times 2$) | LOD 2 ($4\times 4\times 4$) |
|---|---|---|---|
| **Occupancy** | $1.0$ if solid, else $0.0$ | $\frac{\text{solid count}}{8}$ | $\frac{\text{solid count}}{64}$ |
| **Passability** | Exact block passability ($0.0$, $0.5$, or $1.0$) | Mean of 8 sub-blocks | Mean of 64 sub-blocks |
| **Hazard** | $1.0$ if hazard, else $0.0$ | $1.0$ if $\ge 1$ hazard sub-block | $1.0$ if $\ge 1$ hazard sub-block |
| **Hardness** | Exact block hardness | Mean of 8 sub-blocks | Mean of 64 sub-blocks |
| **Light Level** | Exact light $/ 15.0$ | Mean of 8 sub-blocks | Mean of 64 sub-blocks |
| **Fluid Flow** | Exact egocentric $[f_x, f_z]$ | Mean of fluid sub-blocks | Mean of fluid sub-blocks |

---

## 5. Performance Engineering

Extracting 6 channels across up to 262,144 blocks per snapshot must not degrade client performance:

1. **Zero-Allocation Primitive Accumulator**:
   - `PhysicalPropertyExtractor.Accumulator` uses primitive scalar fields (`int solidCount`, `float passabilitySum`, `boolean hasHazard`, etc.).
   - A single accumulator instance is created once per sampling run and `.reset()` is called between cells.
2. **Fast-Path Passability Lookup**:
   - Air directly returns $1.0$.
   - Pre-hashed sets (`IMPEDING_BLOCKS`, `PASSABLE_FLORA`) and `BlockPriorityRegistry` resolve $99\%$ of blocks without expensive voxel collision shape raycasts.
3. **Single Traversal Pass**:
   - Both the categorical block priority (Task 7) and all physical channels (Task 8) are extracted within the **same single loop pass** over the sub-volume.

---

## 6. PyTorch Tensor Conversion Example

```python
import json
import torch
import numpy as np

def observation_to_tensors(obs_json_path="observations/obs_latest.json"):
    with open(obs_json_path, "r") as f:
        data = json.load(f)

    grid = data["grid"] # 16x16x16 list of cell dicts
    size = 16

    # Channels: 0: occupancy, 1: passability, 2: hazard, 3: hardness, 4: light, 5: flowX, 6: flowZ
    channels = 7
    tensor = np.zeros((channels, size, size, size), dtype=np.float32)

    for gx in range(size):
        for gy in range(size):
            for gz in range(size):
                cell = grid[gx][gy][gz]
                props = cell["properties"]
                tensor[0, gx, gy, gz] = props["occupancy"]
                tensor[1, gx, gy, gz] = props["passability"]
                tensor[2, gx, gy, gz] = props["hazard"]
                tensor[3, gx, gy, gz] = props["hardness"]
                tensor[4, gx, gy, gz] = props["lightLevel"]
                tensor[5, gx, gy, gz] = props["fluidFlow"][0]
                tensor[6, gx, gy, gz] = props["fluidFlow"][1]

    # Convert to PyTorch float tensor for 3D CNN: shape [Batch, Channels, D, H, W]
    torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
    print(f"Constructed PyTorch tensor: {torch_tensor.shape}")
    return torch_tensor

if __name__ == "__main__":
    observation_to_tensors()
```

