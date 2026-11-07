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
* If a block is completely surrounded by opaque blocks, set its visibility channel to $0.0$ and assign it the `MASKED` token so the agent does not observe underground caves through solid stone.



---

### Phase 6: Serialization & Pipeline Optimization

*Transition from debug logging to fast data transfer.*

13. **Binary Buffer Serialization (Replacing JSON):**
* JSON text generation is too slow for 20-tick RL loops.
* Serialize the tensors into direct byte arrays (`ByteBuffer` in Java) storing flat float32 or uint8 arrays shaped `[channels, 16, 16, 16]`.
* Keep `/squish` as a debug command for exporting JSON samples, but implement a direct socket, named pipe, or memory-mapped file bridge for sending raw buffers directly into PyTorch.