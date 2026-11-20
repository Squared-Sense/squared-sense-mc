"""
Shared Memory (MMF) Binary Observation Emitter.
Writes offline extracted observation tensors directly into the shared memory IPC bus
(Protocol v3), enabling real-time inspection in the Three.js 3D visualizer without
needing a running Minecraft client.
"""

import mmap
import os
import platform
import struct
import sys
import time
from typing import Dict, Optional, Tuple
import numpy as np

# Add project root and bridge to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
if BRIDGE_DIR not in sys.path:
    sys.path.append(BRIDGE_DIR)

import mc_ipc_layout as layout

CARDINAL_MAP = {
    "SOUTH": 0,
    "WEST": 1,
    "NORTH": 2,
    "EAST": 3
}


def resolve_shm_path(custom_path: Optional[str] = None) -> str:
    if custom_path:
        return custom_path
    system = platform.system().lower()
    if "linux" in system and os.path.exists("/dev/shm"):
        return "/dev/shm/squared_sense_mc.bin"
    temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
    return os.path.join(temp_dir, "squared_sense_mc.bin")


class MinecraftIpcEmitter:
    """
    Lock-free Ping-Pong double-buffered MMF writer conforming to Protocol v3.
    """

    def __init__(self, shm_path: Optional[str] = None):
        self.shm_path = resolve_shm_path(shm_path)
        self.file = None
        self.mmap_obj = None
        self.active_slot = 0
        self.sequence_id = 0

    def open(self):
        """Initializes or opens the memory-mapped file with correct Protocol v3 file size."""
        total_size = layout.TOTAL_FILE_SIZE  # 1,093,632 bytes
        os.makedirs(os.path.dirname(os.path.abspath(self.shm_path)), exist_ok=True)

        # Open in r+b or create if missing
        if not os.path.exists(self.shm_path) or os.path.getsize(self.shm_path) != total_size:
            with open(self.shm_path, "wb") as f:
                f.write(b"\x00" * total_size)

        self.file = open(self.shm_path, "r+b")
        self.mmap_obj = mmap.mmap(self.file.fileno(), total_size, access=mmap.ACCESS_WRITE)
        print(f"[IpcEmitter] Opened shared memory channel at: {self.shm_path} ({total_size:,} bytes)")

    def close(self):
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                pass
            self.mmap_obj = None
        if self.file:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None

    def emit_observation(self, obs: Dict, game_tick: int = 0):
        """
        Writes an observation dictionary into the inactive buffer slot and flips the active slot.
        """
        if self.mmap_obj is None:
            self.open()

        header_data = obs["header"]
        blocks = obs["blocks"]
        entities = obs["entities"]

        # 1. Select inactive slot for writing
        target_slot = 1 - self.active_slot
        slot_offset = layout.get_slot_offset(target_slot, version=layout.PROTOCOL_VERSION)

        # 2. Write Block Tensors (B0, B1, B2)
        for lod in range(3):
            lod_key = f"lod{lod}"
            b = blocks[lod_key]
            lod_block_base = layout.get_lod_block_offset(slot_offset, lod, version=layout.PROTOCOL_VERSION)

            # Block ID (uint16)
            self._write_array(b["block_id"].astype("<u2"), lod_block_base + layout.OFF_BLOCK_ID)
            # Occupancy (float32)
            self._write_array(b["occupancy"].astype("<f4"), lod_block_base + layout.OFF_OCCUPANCY)
            # Passability (float32)
            self._write_array(b["passability"].astype("<f4"), lod_block_base + layout.OFF_PASSABILITY)
            # Hazard (float32)
            self._write_array(b["hazard"].astype("<f4"), lod_block_base + layout.OFF_HAZARD)
            # Hardness (float32)
            self._write_array(b["hardness"].astype("<f4"), lod_block_base + layout.OFF_HARDNESS)
            # Light level (float32)
            self._write_array(b["light_level"].astype("<f4"), lod_block_base + layout.OFF_LIGHT)
            # Fluid Flow X (float32)
            self._write_array(b["fluid_flow_x"].astype("<f4"), lod_block_base + layout.OFF_FLUID_X)
            # Fluid Flow Z (float32)
            self._write_array(b["fluid_flow_z"].astype("<f4"), lod_block_base + layout.OFF_FLUID_Z)
            # Visibility (float32)
            self._write_array(b["visibility"].astype("<f4"), lod_block_base + layout.OFF_VISIBILITY)

        # 3. Write Entity Tensors (E0, E1, E2)
        for lod in range(3):
            lod_key = f"lod{lod}"
            e = entities[lod_key]
            lod_entity_base = layout.get_lod_entity_offset(slot_offset, lod, version=layout.PROTOCOL_VERSION)

            self._write_array(e["entity_count"].astype("<u1"), lod_entity_base + layout.OFF_ENTITY_COUNT)
            self._write_array(e["entity_type_id"].astype("<u1"), lod_entity_base + layout.OFF_ENTITY_TYPE_ID)
            self._write_array(e["is_hostile"].astype("<f4"), lod_entity_base + layout.OFF_IS_HOSTILE)
            self._write_array(e["total_health"].astype("<f4"), lod_entity_base + layout.OFF_TOTAL_HEALTH)

        # 4. Commit and Flip Slot via Header Update
        self.sequence_id += 1
        pos = header_data["playerPos"]
        cardinal_str = header_data.get("cardinal", "SOUTH")
        cardinal_idx = CARDINAL_MAP.get(cardinal_str, 0)
        flags = layout.FLAG_STREAM_ACTIVE
        if header_data.get("visibilityFilter", True):
            flags |= layout.FLAG_VISIBILITY_FILTER

        strides = header_data.get("strides", (1, 2, 4))

        # Pack header struct (96 bytes, Little-Endian)
        # Format: < I I I I Q I I q q d d d f f f B B B B I B B B B
        header_bytes = struct.pack(
            "<IIIIQIIqqdddfffBBBBIBBBB",
            layout.MAGIC,                     # magic uint32
            layout.PROTOCOL_VERSION,          # version uint32
            layout.HEADER_SIZE,               # headerSize uint32
            layout.SLOT_SIZE,                 # slotSize uint32
            self.sequence_id,                 # sequenceId uint64
            target_slot,                      # activeSlot uint32
            flags,                            # flags uint32
            game_tick,                        # gameTick int64
            time.time_ns(),                   # timestampNanos int64
            pos[0], pos[1], pos[2],           # playerX, playerY, playerZ (3 doubles)
            header_data.get("cameraYaw", 0.0),
            header_data.get("cameraPitch", 0.0),
            header_data.get("snappedYaw", 0.0),
            cardinal_idx,                     # cardinal uint8
            header_data.get("footIndex", 6),  # footIndex uint8
            header_data.get("forwardIndex", 2), # forwardIndex uint8
            header_data.get("lateralIndex", 8), # lateralIndex uint8
            header_data.get("vocabSize", 1059), # vocabSize uint32
            strides[0], strides[1], strides[2], 0 # strides + reserved (4 uint8)
        )

        self.mmap_obj.seek(layout.HEADER_OFFSET)
        self.mmap_obj.write(header_bytes)
        self.mmap_obj.flush()

        self.active_slot = target_slot

    def _write_array(self, arr: np.ndarray, offset: int):
        raw = arr.tobytes()
        self.mmap_obj.seek(offset)
        self.mmap_obj.write(raw)

