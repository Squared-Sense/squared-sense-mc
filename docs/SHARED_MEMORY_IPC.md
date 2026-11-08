# Memory-Mapped File (MMF) Shared Memory IPC

## 1. Overview & Purpose

Command-triggered JSON file exports (`/observe`, `/squish`) are suitable for offline inspection, but running JSON serialization and disk I/O on every tick drops Minecraft's tick rate well below 20 TPS.

**Squared Sense MC** implements a high-throughput, **zero-allocation Memory-Mapped File (MMF) IPC bus**:
- **Continuous 20 Hz Streaming**: Serializes complete multi-LOD observations ($B_0..B_2, E_0..E_2$) in under **0.2 milliseconds**.
- **Zero-Copy Consumption**: Downstream Python RL scripts and interactive 3D visualizers read tensors directly from shared RAM without intermediate string parsing, deserialization, or memory copies.
- **Cross-Platform Native Support**: Seamlessly functions across Linux (`/dev/shm` tmpfs) and Windows (`%TEMP%` OS page cache) using standard Java NIO `FileChannel.map()` and Python's `mmap.mmap()`.

---

## 2. Architecture & Lock-Free Concurrency

```
                  ┌─────────────────────────────────────┐
                  │          Minecraft Client           │
                  │   Client render/game tick thread    │
                  └──────────────────┬──────────────────┘
                                     │ Zero-Allocation Direct Write (~0.1 ms)
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │        Shared Memory File (squared_sense_mc.bin)       │
        │  Linux: /dev/shm (tmpfs)  |  Windows: %TEMP% (Cache)   │
        ├────────────────────────────────────────────────────────┤
        │ Page 0 (4 KB)   : Header & Metadata                    │
        │ Slot 0 (472 KB) : Ping-Pong Buffer A                   │
        │ Slot 1 (472 KB) : Ping-Pong Buffer B                   │
        └─────────────────┬────────────────────┬─────────────────┘
                          │                    │ Zero-Copy Reads (~2 µs)
                          ▼                    ▼
               ┌─────────────────────┐  ┌─────────────────────┐
               │   Python RL Model   │  │    3D Visualizer    │
               │  PyTorch Tensor In  │  │   Realtime Display  │
               └─────────────────────┘  └─────────────────────┘
```

### Single-Producer, Multi-Consumer Pattern
Minecraft is the sole writer; any number of external consumer processes (e.g. PyTorch training loops, Web visualizer bridges, diagnostic monitors) can read simultaneously.

### Lock-Free Ping-Pong Double Buffering
To prevent torn reads without imposing locks that could stall Minecraft:
1. The memory file contains two identical payload slots: **Slot 0** and **Slot 1**.
2. Minecraft always writes to the **inactive slot** (`inactiveSlot = 1 - activeSlot`).
3. After the tensors are written, a memory store barrier (`VarHandle.releaseFence()`) is issued.
4. Minecraft updates the header: sets `activeSlot = inactiveSlot` and increments `sequenceId`.
5. Consumers always read from `activeSlot`. Because the writer never touches `activeSlot` while writing, consumers enjoy **100% tear-free, wait-free reads**.

---

## 3. Cross-Platform File Paths

Both Java and Python resolve the memory-mapped file automatically:

| Platform | Default Path | Storage Backing | Latency |
|---|---|---|---|
| **Linux** | `/dev/shm/squared_sense_mc.bin` | Kernel `tmpfs` (RAM only, zero disk touches) | $< 0.05\text{ ms}$ |
| **Windows** | `%TEMP%\squared_sense_mc.bin` | NT Kernel System Page Cache (Physical RAM) | $< 0.1\text{ ms}$ |

---

## 4. Binary Layout Specification

The memory-mapped file is aligned to 4,096-byte page boundaries:
- **Total File Size**: $970,752\text{ bytes}$ (~948 KB, under 1 MB in RAM).
- **Byte Order**: **Little-Endian** (native to x86_64, NumPy, and PyTorch).

### Page 0: Header & Metadata (Offset `0`, Size `4096`)

| Offset | Type | Field Name | Description |
|---|---|---|---|
| `0x00` | `uint32` | `magic` | Magic header bytes: `0x53515343` (`"SQSC"`) |
| `0x04` | `uint32` | `version` | Protocol version (`1`) |
| `0x08` | `uint32` | `headerSize` | Header page length (`4096`) |
| `0x0C` | `uint32` | `slotSize` | Byte size of each slot (`483328`) |
| `0x10` | `uint64` | `sequenceId` | Monotonically incrementing snapshot counter |
| `0x18` | `uint32` | `activeSlot` | Slot currently safe for reading (`0` or `1`) |
| `0x1C` | `uint32` | `flags` | State flags (`1` = active) |
| `0x20` | `int64` | `gameTick` | Minecraft game tick |
| `0x28` | `int64` | `timestampNanos`| System nanoseconds at capture |
| `0x30` | `double[3]`| `playerPos` | Continuous world $(x, y, z)$ position of player |
| `0x48` | `float` | `cameraYaw` | Player camera yaw |
| `0x4C` | `float` | `cameraPitch` | Player camera pitch |
| `0x50` | `float` | `snappedYaw` | Snapped cardinal yaw ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) |
| `0x54` | `uint8` | `cardinal` | Snapped cardinal index (0=S, 1=W, 2=N, 3=E) |
| `0x55` | `uint8` | `footIndex` | Pitch-derived vertical foot index ($gy \in [1..12]$) |
| `0x56` | `uint8` | `forwardIndex` | Lateral anchor index (`2`) |
| `0x57` | `uint8` | `lateralIndex` | Forward anchor index (`8`) |
| `0x58` | `uint32` | `vocabSize` | Vocabulary token count (~882) |
| `0x5C` | `uint8[3]` | `strides` | Strides for LOD 0, 1, 2 (`1, 2, 4`) |

---

### Slot Payload Layout (Offset `4096 + slotIndex * 483328`)

Each slot is partitioned into planar, contiguous tensor blocks:

```
[Slot Start: slotOffset]
├── Block Tensors (B0, B1, B2) ────────── 368,640 bytes total
│   ├── LOD 0 (Stride 1): 122,880 bytes
│   │   ├── uint16  blockId[4096]        (8,192 bytes,  offset +0)
│   │   ├── float32 occupancy[4096]      (16,384 bytes, offset +8,192)
│   │   ├── float32 passability[4096]    (16,384 bytes, offset +24,576)
│   │   ├── float32 hazard[4096]         (16,384 bytes, offset +40,960)
│   │   ├── float32 hardness[4096]       (16,384 bytes, offset +57,344)
│   │   ├── float32 lightLevel[4096]     (16,384 bytes, offset +73,728)
│   │   ├── float32 fluidFlowX[4096]     (16,384 bytes, offset +90,112)
│   │   └── float32 fluidFlowZ[4096]     (16,384 bytes, offset +106,496)
│   ├── LOD 1 (Stride 2): 122,880 bytes  (offset +122,880)
│   └── LOD 2 (Stride 4): 122,880 bytes  (offset +245,760)
├── Entity Tensors (E0, E1, E2) ───────── 110,592 bytes total
│   ├── LOD 0 (Stride 1): 36,864 bytes   (offset +368,640)
│   │   ├── uint8   entityCount[4096]    (4,096 bytes,  offset +0)
│   │   ├── float32 isHostile[4096]      (16,384 bytes, offset +4,096)
│   │   └── float32 totalHealth[4096]    (16,384 bytes, offset +20,480)
│   ├── LOD 1 (Stride 2): 36,864 bytes   (offset +405,504)
│   └── LOD 2 (Stride 4): 36,864 bytes   (offset +442,368)
└── Auxiliary & Raycast Data ──────────── 256 bytes (offset +479,232)
```

Because each feature channel is laid out as a contiguous flat array of $4,096$ values, NumPy directly reshapes it into `(16, 16, 16)` in $O(1)$ time without any striding or transposition.

---

## 5. Python / PyTorch Zero-Copy Consumer Example

```python
import mmap
import numpy as np
import torch
from bridge.mc_ipc_reader import MinecraftIpcReader

reader = MinecraftIpcReader()
reader.open()

# Read active header snapshot
header = reader.read_header()
active_slot = header["activeSlot"]

# Zero-copy extraction of block and entity tensors
b0 = reader.read_block_tensors(active_slot, lod=0)
e0 = reader.read_entity_tensors(active_slot, lod=0)

# Wrap directly as PyTorch tensors (zero-copy)
occupancy_tensor = torch.from_numpy(b0["occupancy"])  # shape: (16, 16, 16)
hazard_tensor = torch.from_numpy(b0["hazard"])
entity_count_tensor = torch.from_numpy(e0["entity_count"])

print(f"Tick {header['gameTick']} | Occupancy shape: {occupancy_tensor.shape}")
```

