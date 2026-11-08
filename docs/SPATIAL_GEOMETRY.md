# Spatial Geometry & Coordinate Frame

This document details the mathematical and geometric foundation of the observational reference frame implemented in **Phase 1** of the observation pipeline.

---

## 1. Relative Coordinates $(\Delta x, \Delta y, \Delta z)$ in 1-Meter Block Units

In observational data for RL:
- **No Absolute World Coordinates**: Absolute coordinates (e.g. `x: 1420, y: 64, z: -3200`) are replaced with real-world block distance relative to the player's feet position:
  $$\Delta x = x_{\text{world}} - x_{\text{player}}$$
  $$\Delta y = y_{\text{world}} - y_{\text{player}}$$
  $$\Delta z = z_{\text{world}} - z_{\text{player}}$$
- **Preserved 1-Meter Units Across All LODs**: Coordinates are **not** scaled down by the LOD stride. If a location is $4$ meters away, its relative coordinate is $\pm 4$ across LOD 0, LOD 1, and LOD 2. This guarantees that matching physical locations share identical spatial feature values.

---

## 2. Egocentric Coordinate Frame

Observations are formatted as a 3D tensor `grid[gx][gy][gz]` of dimensions $16 \times 16 \times 16$:
- **Dimension 0 ($gx$ — Lateral Axis)**: Runs from **Left to Right** across 16 cells.
- **Dimension 1 ($gy$ — Vertical Axis)**: Runs from **Bottom to Top** across 16 cells.
- **Dimension 2 ($gz$ — Forward Axis)**: Runs from **Behind to In-Front** across 16 cells.

This egocentric layout ensures that the RL neural network always interprets:
- $+gz$ as "forward"
- $+gx$ as "right"
- $+gy$ as "up"

---

## 3. Asymmetric Forward / Lateral Offsets (Task 1.2)

In gameplay, an agent rarely needs deep vision behind its back, but requires substantial preview distance ahead.

### Forward Axis ($gz$)
- The player's feet sit at index **`gz = 2`**:
  - $gz = 0$: $2$ cells behind ($z_{\text{local}} = -2 \times \text{stride}$)
  - $gz = 1$: $1$ cell behind ($z_{\text{local}} = -1 \times \text{stride}$)
  - $gz = 2$: Player's position ($z_{\text{local}} = 0$)
  - $gz = 3\dots 15$: $13$ cells in front ($z_{\text{local}} = +1 \dots +13 \times \text{stride}$)
- **Total Reach**: 2 cells behind, 14 cells in front (including the player's cell).

### Lateral Axis ($gx$)
- Symmetrically centered at index **`gx = 8`**:
  - Covered range: indices $[-8, +7]$ relative to the player.
  - $x_{\text{local}} = (gx - 8) \times \text{stride}$.

---

## 4. Cardinal Yaw Snapping with Hysteresis (Task 1.3)

To ensure the grid axes align with Minecraft's discrete voxel grid while remaining rotation-aware, the player's continuous yaw is snapped to one of the 4 cardinal directions:

| Cardinal | Snapped Yaw | Forward Unit Vector $\vec{F}$ | Right Unit Vector $\vec{R}$ |
|---|---|---|---|
| **SOUTH** | $0^\circ$ | $(0, +1)$ ($+Z$) | $(-1, 0)$ ($-X$) |
| **WEST** | $90^\circ$ | $(-1, 0)$ ($-X$) | $(0, -1)$ ($-Z$) |
| **NORTH** | $180^\circ$ | $(0, -1)$ ($-Z$) | $(+1, 0)$ ($+X$) |
| **EAST** | $270^\circ$ | $(+1, 0)$ ($+X$) | $(0, +1)$ ($+Z$) |

### Hysteresis Deadband ($\pm 10^\circ$)
Nominal diagonal boundaries sit at $45^\circ, 135^\circ, 225^\circ, 315^\circ$. To prevent rapid grid oscillation ("thrashing") when the player looks along diagonal directions, a **$\pm 10^\circ$ deadband** is enforced:

- **South $\leftrightarrow$ West ($45^\circ$)**:
  - Switch `SOUTH -> WEST` only when $\text{yaw} \ge 55^\circ$.
  - Switch `WEST -> SOUTH` only when $\text{yaw} < 35^\circ$.
  - Between $35^\circ$ and $55^\circ$, the existing snapped direction is maintained.
- **West $\leftrightarrow$ North ($135^\circ$)**:
  - Switch `WEST -> NORTH` only when $\text{yaw} \ge 145^\circ$.
  - Switch `NORTH -> WEST` only when $\text{yaw} < 125^\circ$.
- **North $\leftrightarrow$ East ($225^\circ$)**:
  - Switch `NORTH -> EAST` only when $\text{yaw} \ge 235^\circ$.
  - Switch `EAST -> NORTH` only when $\text{yaw} < 215^\circ$.
- **East $\leftrightarrow$ South ($315^\circ$)**:
  - Switch `EAST -> SOUTH` only when $\text{yaw} \ge 325^\circ$.
  - Switch `SOUTH -> EAST` only when $\text{yaw} < 305^\circ$.

---

## 5. Vertical Pitch Translation (Task 1.4)

When looking down (e.g. mining, parkour), blocks beneath the player are critical; when looking up (e.g. building, climbing), overhead blocks matter. The vertical sampling window dynamically shifts based on head pitch $\phi \in [-90^\circ, +90^\circ]$:

- **Looking Straight Down ($\phi = +90^\circ$)**:
  - Foot level sits at **`footIndex = 12`**.
  - Exposes **12 blocks below** the player's feet ($gy = 0\dots 11$) and 3 blocks above ($gy = 13\dots 15$).
- **Looking Horizontal ($\phi = 0^\circ$)**:
  - Foot level sits at **`footIndex = 6`**.
  - Exposes **6 blocks below** ($gy = 0\dots 5$) and 9 blocks above ($gy = 7\dots 15$).
- **Looking Straight Up ($\phi = -90^\circ$)**:
  - Foot level sits at **`footIndex = 1`**.
  - Exposes **1 block below** ($gy = 0$) and 14 blocks above ($gy = 2\dots 15$).

### Formula
$$y_{\text{local}} = (gy - \text{footIndex}) \times \text{stride}$$

Where $\text{footIndex}$ is computed as:
$$\text{footIndex} = \begin{cases} 
\text{round}\left(6 + \frac{\phi}{90} \times (12 - 6)\right) = \text{round}\left(6 + 6 \frac{\phi}{90}\right), & \phi \ge 0 \\
\text{round}\left(6 - \frac{-\phi}{90} \times (6 - 1)\right) = \text{round}\left(6 + 5 \frac{\phi}{90}\right), & \phi < 0
\end{cases}$$

---

## 6. Complete Coordinate Transformation Pipeline

For any cell $(gx, gy, gz) \in [0..15]^3$:

1. **Compute Local Offsets**:
   $$x_{\text{local}} = (gx - 8) \times \text{stride}$$
   $$y_{\text{local}} = (gy - \text{footIndex}) \times \text{stride}$$
   $$z_{\text{local}} = (gz - 2) \times \text{stride}$$

2. **Rotate into World Relative Delts $(\Delta x, \Delta y, \Delta z)$**:
   $$\Delta x = x_{\text{local}} \cdot R_x + z_{\text{local}} \cdot F_x$$
   $$\Delta y = y_{\text{local}}$$
   $$\Delta z = x_{\text{local}} \cdot R_z + z_{\text{local}} \cdot F_z$$

3. **Query Client World Position**:
   $$x_{\text{world}} = x_{\text{player}} + \Delta x$$
   $$y_{\text{world}} = y_{\text{player}} + \Delta y$$
   $$z_{\text{world}} = z_{\text{player}} + \Delta z$$

