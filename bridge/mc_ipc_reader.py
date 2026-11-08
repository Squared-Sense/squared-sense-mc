"""
Zero-Copy Memory-Mapped File (MMF) Reader for Squared Sense MC.
Reads live observation tensors directly from shared memory at sub-millisecond latency.
"""

import mmap
import os
import platform
import struct
import time
from typing import Dict, Optional, Tuple
import numpy as np

import mc_ipc_layout as layout

CARDINAL_NAMES = ["SOUTH", "WEST", "NORTH", "EAST"]


def resolve_default_shm_path() -> str:
    system = platform.system().lower()
    if "linux" in system and os.path.exists("/dev/shm"):
        return "/dev/shm/squared_sense_mc.bin"
    temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
    return os.path.join(temp_dir, "squared_sense_mc.bin")


class MinecraftIpcReader:
    def __init__(self, shm_path: Optional[str] = None):
        self.shm_path = shm_path or resolve_default_shm_path()
        self.file = None
        self.mmap_obj = None

    def open(self) -> bool:
        if not os.path.exists(self.shm_path):
            return False
        try:
            self.file = open(self.shm_path, "r+b")
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            return True
        except Exception:
            self.close()
            return False

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

    def is_open(self) -> bool:
        return self.mmap_obj is not None

    def ensure_current_mapping(self) -> bool:
        """Dynamically reopens the MMF if file size changed on disk (e.g. layout version upgrade)."""
        if not os.path.exists(self.shm_path):
            return False
        try:
            current_size = os.path.getsize(self.shm_path)
            if current_size == 0:
                return False
            if self.mmap_obj is None or len(self.mmap_obj) != current_size:
                self.close()
                return self.open()
            return True
        except Exception:
            return False

    def _safe_frombuffer(self, dtype, count: int, offset: int) -> np.ndarray:
        """Reads a slice of the mmap buffer safely, remapping if necessary and preventing ValueError."""
        itemsize = np.dtype(dtype).itemsize
        needed = offset + count * itemsize
        if not self.mmap_obj or len(self.mmap_obj) < needed:
            self.ensure_current_mapping()
        if not self.mmap_obj or len(self.mmap_obj) < needed:
            return np.zeros(count, dtype=dtype)
        return np.frombuffer(self.mmap_obj, dtype=dtype, count=count, offset=offset)

    def read_header(self) -> Optional[Dict]:
        if not self.ensure_current_mapping():
            return None

        self.mmap_obj.seek(layout.HEADER_OFFSET)
        header_bytes = self.mmap_obj.read(96)
        if len(header_bytes) < 96:
            return None

        # Unpack header struct (Little-Endian)
        # 0: magic (uint32)
        # 4: version (uint32)
        # 8: headerSize (uint32)
        # 12: slotSize (uint32)
        # 16: sequenceId (uint64)
        # 24: activeSlot (uint32)
        # 28: flags (uint32)
        # 32: gameTick (int64)
        # 40: timestampNanos (int64)
        # 48: playerX, Y, Z (3 x double = 24)
        # 72: cameraYaw, pitch, snappedYaw (3 x float = 12)
        # 84: cardinal, footIndex, forwardIndex, lateralIndex (4 x uint8 = 4)
        # 88: vocabSize (uint32 = 4)
        # 92: stride0, stride1, stride2, reserved (4 x uint8 = 4)
        unpacked = struct.unpack("<IIIIQIIqqdddfffBBBBIBBBB", header_bytes)

        magic = unpacked[0]
        if magic != layout.MAGIC:
            return None

        cardinal_idx = unpacked[15]
        return {
            "version": unpacked[1],
            "headerSize": unpacked[2],
            "slotSize": unpacked[3],
            "sequenceId": unpacked[4],
            "activeSlot": unpacked[5],
            "flags": unpacked[6],
            "gameTick": unpacked[7],
            "timestampNanos": unpacked[8],
            "playerPos": (unpacked[9], unpacked[10], unpacked[11]),
            "cameraYaw": unpacked[12],
            "cameraPitch": unpacked[13],
            "snappedYaw": unpacked[14],
            "cardinal": CARDINAL_NAMES[cardinal_idx] if cardinal_idx < 4 else "UNKNOWN",
            "footIndex": unpacked[16],
            "forwardIndex": unpacked[17],
            "lateralIndex": unpacked[18],
            "vocabSize": unpacked[19],
            "strides": (unpacked[20], unpacked[21], unpacked[22])
        }

    def get_protocol_version(self) -> int:
        """Reads protocol version directly from shared memory header."""
        if self.mmap_obj is None or len(self.mmap_obj) < 8:
            return layout.PROTOCOL_VERSION
        self.mmap_obj.seek(layout.HEADER_OFFSET + layout.OFF_VERSION)
        ver_bytes = self.mmap_obj.read(4)
        if len(ver_bytes) == 4:
            return struct.unpack("<I", ver_bytes)[0]
        return layout.PROTOCOL_VERSION

    def read_block_tensors(self, active_slot: int, lod: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts zero-copy NumPy 3D arrays for all channels in the specified LOD (0, 1, or 2).
        Shape: [16, 16, 16] for each channel (gx, gy, gz).
        """
        assert 0 <= lod < 3, "LOD must be 0, 1, or 2"
        version = self.get_protocol_version()
        slot_offset = layout.get_slot_offset(active_slot, version=version)
        lod_base = layout.get_lod_block_offset(slot_offset, lod)

        n = layout.CELLS_PER_LOD  # 4096
        shape = (layout.GRID_SIZE, layout.GRID_SIZE, layout.GRID_SIZE)

        block_id = self._safe_frombuffer(
            np.uint16, n, lod_base + layout.OFF_BLOCK_ID
        ).reshape(shape)

        occupancy = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_OCCUPANCY
        ).reshape(shape)

        passability = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_PASSABILITY
        ).reshape(shape)

        hazard = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_HAZARD
        ).reshape(shape)

        hardness = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_HARDNESS
        ).reshape(shape)

        light_level = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_LIGHT
        ).reshape(shape)

        fluid_flow_x = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_FLUID_X
        ).reshape(shape)

        fluid_flow_z = self._safe_frombuffer(
            np.float32, n, lod_base + layout.OFF_FLUID_Z
        ).reshape(shape)

        return {
            "block_id": block_id,
            "occupancy": occupancy,
            "passability": passability,
            "hazard": hazard,
            "hardness": hardness,
            "light_level": light_level,
            "fluid_flow_x": fluid_flow_x,
            "fluid_flow_z": fluid_flow_z
        }

    def read_entity_tensors(self, active_slot: int, lod: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts zero-copy NumPy 3D arrays for entity channels in the specified LOD.
        Shape: [16, 16, 16] (gx, gy, gz).
        Supports both Protocol v1 (legacy) and Protocol v2 (with entity_type_id).
        """
        assert 0 <= lod < 3, "LOD must be 0, 1, or 2"
        version = self.get_protocol_version()
        slot_offset = layout.get_slot_offset(active_slot, version=version)
        lod_base = layout.get_lod_entity_offset(slot_offset, lod, version=version)

        n = layout.CELLS_PER_LOD
        shape = (layout.GRID_SIZE, layout.GRID_SIZE, layout.GRID_SIZE)

        entity_count = self._safe_frombuffer(
            np.uint8, n, lod_base + layout.OFF_ENTITY_COUNT
        ).reshape(shape)

        if version >= 2:
            entity_type_id = self._safe_frombuffer(
                np.uint8, n, lod_base + layout.OFF_ENTITY_TYPE_ID
            ).reshape(shape)

            is_hostile = self._safe_frombuffer(
                np.float32, n, lod_base + layout.OFF_IS_HOSTILE
            ).reshape(shape)

            total_health = self._safe_frombuffer(
                np.float32, n, lod_base + layout.OFF_TOTAL_HEALTH
            ).reshape(shape)
        else:
            entity_type_id = np.zeros(shape, dtype=np.uint8)

            is_hostile = self._safe_frombuffer(
                np.float32, n, lod_base + layout.V1_OFF_IS_HOSTILE
            ).reshape(shape)

            total_health = self._safe_frombuffer(
                np.float32, n, lod_base + layout.V1_OFF_TOTAL_HEALTH
            ).reshape(shape)

        return {
            "entity_count": entity_count,
            "entity_type_id": entity_type_id,
            "is_hostile": is_hostile,
            "total_health": total_health
        }


def main():
    reader = MinecraftIpcReader()
    print(f"Connecting to MMF at: {reader.shm_path}...")

    while not reader.open():
        print("Waiting for Minecraft mod to initialize MMF channel... (retrying in 1s)")
        time.sleep(1.0)

    print("Connected! Streaming live observations...")
    last_seq = -1
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            header = reader.read_header()
            if header and header["sequenceId"] != last_seq:
                last_seq = header["sequenceId"]
                frame_count += 1
                slot = header["activeSlot"]

                # Fast zero-copy tensor extraction
                b0 = reader.read_block_tensors(slot, lod=0)
                e0 = reader.read_entity_tensors(slot, lod=0)

                solid_count = np.count_nonzero(b0["occupancy"] > 0)
                entity_total = np.sum(e0["entity_count"])

                fps = frame_count / max(0.001, time.time() - start_time)
                pos = header["playerPos"]
                print(
                    f"\r[Seq {last_seq:06d} | Tick {header['gameTick']:06d} | Slot {slot}] "
                    f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                    f"Facing: {header['cardinal']} | Solid: {solid_count:4d}/4096 | Entities: {entity_total} "
                    f"({fps:.1f} Hz)",
                    end="",
                    flush=True
                )
            time.sleep(0.01)  # 100 Hz poll
    except KeyboardInterrupt:
        print("\nStopping reader.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()

