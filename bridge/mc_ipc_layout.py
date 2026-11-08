"""
Memory-Mapped File (MMF) Layout Constants for Squared Sense MC IPC.
Matches org.sqsense.squared_sense_mc.client.observation.ipc.IpcBufferLayout.
"""

MAGIC = 0x53515343  # "SQSC"
PROTOCOL_VERSION = 3

GRID_SIZE = 16
CELLS_PER_LOD = GRID_SIZE * GRID_SIZE * GRID_SIZE  # 4096
NUM_LODS = 3

# Flags (OFF_FLAGS uint32 bitmask)
FLAG_STREAM_ACTIVE = 1 << 0
FLAG_VISIBILITY_FILTER = 1 << 1

# Header Offsets (within Page 0, 4096 bytes)
HEADER_OFFSET = 0
HEADER_SIZE = 4096

OFF_MAGIC = 0               # uint32 (4)
OFF_VERSION = 4             # uint32 (4)
OFF_HEADER_SIZE = 8         # uint32 (4)
OFF_SLOT_SIZE = 12          # uint32 (4)
OFF_SEQUENCE_ID = 16        # uint64 (8)
OFF_ACTIVE_SLOT = 24        # uint32 (4)
OFF_FLAGS = 28              # uint32 (4)
OFF_GAME_TICK = 32          # int64 (8)
OFF_TIMESTAMP_NANOS = 40    # int64 (8)
OFF_PLAYER_X = 48           # double (8)
OFF_PLAYER_Y = 56           # double (8)
OFF_PLAYER_Z = 64           # double (8)
OFF_CAMERA_YAW = 72         # float (4)
OFF_CAMERA_PITCH = 76       # float (4)
OFF_SNAPPED_YAW = 80        # float (4)
OFF_CARDINAL = 84           # uint8 (1)
OFF_FOOT_INDEX = 85         # uint8 (1)
OFF_FORWARD_INDEX = 86      # uint8 (1)
OFF_LATERAL_INDEX = 87      # uint8 (1)
OFF_VOCAB_SIZE = 88         # uint32 (4)
OFF_STRIDE_0 = 92           # uint8 (1)
OFF_STRIDE_1 = 93           # uint8 (1)
OFF_STRIDE_2 = 94           # uint8 (1)

# Block Channel Offsets inside each LOD block section
OFF_BLOCK_ID = 0
OFF_OCCUPANCY = 8192
OFF_PASSABILITY = 24576
OFF_HAZARD = 40960
OFF_HARDNESS = 57344
OFF_LIGHT = 73728
OFF_FLUID_X = 90112
OFF_FLUID_Z = 106496
OFF_VISIBILITY = 122880

BLOCK_BYTES_PER_LOD = CELLS_PER_LOD * 34  # 139,264
TOTAL_BLOCK_BYTES = BLOCK_BYTES_PER_LOD * NUM_LODS  # 417,792

# Entity Channel Offsets inside each LOD entity section (Protocol v2 & v3)
OFF_ENTITY_COUNT = 0
OFF_ENTITY_TYPE_ID = 4096
OFF_IS_HOSTILE = 8192
OFF_TOTAL_HEALTH = 24576

ENTITY_BYTES_PER_LOD = CELLS_PER_LOD * 10  # 40,960
TOTAL_ENTITY_BYTES = ENTITY_BYTES_PER_LOD * NUM_LODS  # 122,880

AUX_BYTES = 256
UNPADDED_SLOT_SIZE = TOTAL_BLOCK_BYTES + TOTAL_ENTITY_BYTES + AUX_BYTES
SLOT_SIZE = ((UNPADDED_SLOT_SIZE + 4095) // 4096) * 4096  # 544,768

SLOT_0_OFFSET = HEADER_SIZE
SLOT_1_OFFSET = HEADER_SIZE + SLOT_SIZE
TOTAL_FILE_SIZE = HEADER_SIZE + SLOT_SIZE * 2  # 1,093,632 bytes

# Protocol V2 Fallback Offsets (before Phase 5 visibility channel)
V2_BLOCK_BYTES_PER_LOD = CELLS_PER_LOD * 30  # 122,880
V2_TOTAL_BLOCK_BYTES = V2_BLOCK_BYTES_PER_LOD * NUM_LODS  # 368,640
V2_UNPADDED_SLOT_SIZE = V2_TOTAL_BLOCK_BYTES + TOTAL_ENTITY_BYTES + AUX_BYTES
V2_SLOT_SIZE = ((V2_UNPADDED_SLOT_SIZE + 4095) // 4096) * 4096  # 495,616

# Protocol V1 Fallback Offsets (before Protocol v2 entityTypeId)
V1_OFF_ENTITY_COUNT = 0
V1_OFF_IS_HOSTILE = 4096
V1_OFF_TOTAL_HEALTH = 20480
V1_ENTITY_BYTES_PER_LOD = CELLS_PER_LOD * 9  # 36,864
V1_TOTAL_ENTITY_BYTES = V1_ENTITY_BYTES_PER_LOD * NUM_LODS  # 110,592
V1_UNPADDED_SLOT_SIZE = V2_TOTAL_BLOCK_BYTES + V1_TOTAL_ENTITY_BYTES + AUX_BYTES
V1_SLOT_SIZE = ((V1_UNPADDED_SLOT_SIZE + 4095) // 4096) * 4096  # 483,328


def get_slot_offset(slot_index: int, version: int = PROTOCOL_VERSION) -> int:
    if version >= 3:
        slot_size = SLOT_SIZE
    elif version == 2:
        slot_size = V2_SLOT_SIZE
    else:
        slot_size = V1_SLOT_SIZE
    return HEADER_SIZE + (slot_index & 1) * slot_size


def get_lod_block_offset(slot_offset: int, lod: int, version: int = PROTOCOL_VERSION) -> int:
    block_bytes = BLOCK_BYTES_PER_LOD if version >= 3 else V2_BLOCK_BYTES_PER_LOD
    return slot_offset + lod * block_bytes


def get_lod_entity_offset(slot_offset: int, lod: int, version: int = PROTOCOL_VERSION) -> int:
    block_total = TOTAL_BLOCK_BYTES if version >= 3 else V2_TOTAL_BLOCK_BYTES
    ent_bytes = ENTITY_BYTES_PER_LOD if version >= 2 else V1_ENTITY_BYTES_PER_LOD
    return slot_offset + block_total + lod * ent_bytes


def get_aux_offset(slot_offset: int, version: int = PROTOCOL_VERSION) -> int:
    block_total = TOTAL_BLOCK_BYTES if version >= 3 else V2_TOTAL_BLOCK_BYTES
    total_ent = TOTAL_ENTITY_BYTES if version >= 2 else V1_TOTAL_ENTITY_BYTES
    return slot_offset + block_total + total_ent

