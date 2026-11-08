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

    def read_header(self) -> Optional[Dict]:
        if not self.mmap_obj:
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

        cardinal_idx = unpacked[16]
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
            "footIndex": unpacked[17],
            "forwardIndex": unpacked[18],
            "lateralIndex": unpacked[19],
            "vocabSize": unpacked[20],
            "strides": (unpacked[21], unpacked[22], unpacked[23])
        }

    def read_block_tensors(self, active_slot: int, lod: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts zero-copy NumPy 3D arrays for all channels in the specified LOD (0, 1, or 2).
        Shape: [16, 16, 16] for each channel (gx, gy, gz).
        """
        assert 0 <= lod < 3, "LOD must be 0, 1, or 2"
        slot_offset = layout.get_slot_offset(active_slot)
        lod_base = layout.get_lod_block_offset(slot_offset, lod)

        n = layout.CELLS_PER_LOD  # 4096
        shape = (layout.GRID_SIZE, layout.GRID_SIZE, layout.GRID_SIZE)

        block_id = np.frombuffer(
            self.mmap_obj, dtype=np.uint16, count=n, offset=lod_base + layout.OFF_BLOCK_ID
        ).reshape(shape)

        occupancy = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_OCCUPANCY
        ).reshape(shape)

        passability = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_PASSABILITY
        ).reshape(shape)

        hazard = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_HAZARD
        ).reshape(shape)

        hardness = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_HARDNESS
        ).reshape(shape)

        light_level = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_LIGHT
        ).reshape(shape)

        fluid_flow_x = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_FLUID_X
        ).reshape(shape)

        fluid_flow_z = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_FLUID_Z
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
        """
        assert 0 <= lod < 3, "LOD must be 0, 1, or 2"
        slot_offset = layout.get_slot_offset(active_slot)
        lod_base = layout.get_lod_entity_offset(slot_offset, lod)

        n = layout.CELLS_PER_LOD
        shape = (layout.GRID_SIZE, layout.GRID_SIZE, layout.GRID_SIZE)

        entity_count = np.frombuffer(
            self.mmap_obj, dtype=np.uint8, count=n, offset=lod_base + layout.OFF_ENTITY_COUNT
        ).reshape(shape)

        is_hostile = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_IS_HOSTILE
        ).reshape(shape)

        total_health = np.frombuffer(
            self.mmap_obj, dtype=np.float32, count=n, offset=lod_base + layout.OFF_TOTAL_HEALTH
        ).reshape(shape)

        return {
            "entity_count": entity_count,
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

