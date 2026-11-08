"""
Memory-Mapped File (MMF) Layout Constants for Squared Sense MC IPC.
Matches org.sqsense.squared_sense_mc.client.observation.ipc.IpcBufferLayout.
"""

MAGIC = 0x53515343  # "SQSC"
PROTOCOL_VERSION = 1

GRID_SIZE = 16
CELLS_PER_LOD = GRID_SIZE * GRID_SIZE * GRID_SIZE  # 4096
NUM_LODS = 3

# Header Offsets (within Page 0, 4096 bytes)
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

BLOCK_BYTES_PER_LOD = CELLS_PER_LOD * 30  # 122,880
TOTAL_BLOCK_BYTES = BLOCK_BYTES_PER_LOD * NUM_LODS  # 368,640

# Entity Channel Offsets inside each LOD entity section
OFF_ENTITY_COUNT = 0
OFF_IS_HOSTILE = 4096
OFF_TOTAL_HEALTH = 20480

ENTITY_BYTES_PER_LOD = CELLS_PER_LOD * 9  # 36,864
TOTAL_ENTITY_BYTES = ENTITY_BYTES_PER_LOD * NUM_LODS  # 110,592

AUX_BYTES = 256
UNPADDED_SLOT_SIZE = TOTAL_BLOCK_BYTES + TOTAL_ENTITY_BYTES + AUX_BYTES
SLOT_SIZE = ((UNPADDED_SLOT_SIZE + 4095) // 4096) * 4096  # 483,328

SLOT_0_OFFSET = HEADER_SIZE
SLOT_1_OFFSET = HEADER_SIZE + SLOT_SIZE
TOTAL_FILE_SIZE = HEADER_SIZE + SLOT_SIZE * 2  # 970,752 bytes


def get_slot_offset(slot_index: int) -> int:
    return HEADER_SIZE + (slot_index & 1) * SLOT_SIZE


def get_lod_block_offset(slot_offset: int, lod: int) -> int:
    return slot_offset + lod * BLOCK_BYTES_PER_LOD


def get_lod_entity_offset(slot_offset: int, lod: int) -> int:
    return slot_offset + TOTAL_BLOCK_BYTES + lod * ENTITY_BYTES_PER_LOD

