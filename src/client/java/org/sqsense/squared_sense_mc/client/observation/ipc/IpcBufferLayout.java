package org.sqsense.squared_sense_mc.client.observation.ipc;

/**
 * Byte-level memory layout constants and offset calculators for the Memory-Mapped File (MMF) IPC bus.
 *
 * Total File Layout (Page-aligned, Little-Endian):
 * - Page 0 (4096 bytes): Header & Metadata (128-byte logical struct + padding)
 * - Slot 0 (483,328 bytes): Contiguous tensors for B0..B2, E0..E2, and Aux data
 * - Slot 1 (483,328 bytes): Contiguous tensors for B0..B2, E0..E2, and Aux data
 *
 * Total file size: 970,752 bytes (~948 KB, < 1 MB in RAM).
 */
public final class IpcBufferLayout {
    private IpcBufferLayout() {}

    // Magic & Protocol Version
    public static final int MAGIC = 0x53515343; // "SQSC" in Little-Endian
    public static final int PROTOCOL_VERSION = 3;

    // Flags (OFF_FLAGS uint32 bitmask)
    public static final int FLAG_STREAM_ACTIVE = 1 << 0;
    public static final int FLAG_VISIBILITY_FILTER = 1 << 1;

    // Dimensions
    public static final int GRID_SIZE = 16;
    public static final int CELLS_PER_LOD = GRID_SIZE * GRID_SIZE * GRID_SIZE; // 4096
    public static final int NUM_LODS = 3;

    // Header Offsets (within Page 0, 128 bytes total)
    public static final int HEADER_OFFSET = 0;
    public static final int HEADER_SIZE = 4096; // 4 KB page-aligned

    public static final int OFF_MAGIC = 0;               // uint32 (4)
    public static final int OFF_VERSION = 4;             // uint32 (4)
    public static final int OFF_HEADER_SIZE = 8;         // uint32 (4)
    public static final int OFF_SLOT_SIZE = 12;          // uint32 (4)
    public static final int OFF_SEQUENCE_ID = 16;        // uint64 (8)
    public static final int OFF_ACTIVE_SLOT = 24;        // uint32 (4)
    public static final int OFF_FLAGS = 28;              // uint32 (4)
    public static final int OFF_GAME_TICK = 32;          // int64 (8)
    public static final int OFF_TIMESTAMP_NANOS = 40;    // int64 (8)
    public static final int OFF_PLAYER_X = 48;           // double (8)
    public static final int OFF_PLAYER_Y = 56;           // double (8)
    public static final int OFF_PLAYER_Z = 64;           // double (8)
    public static final int OFF_CAMERA_YAW = 72;         // float (4)
    public static final int OFF_CAMERA_PITCH = 76;       // float (4)
    public static final int OFF_SNAPPED_YAW = 80;        // float (4)
    public static final int OFF_CARDINAL = 84;           // uint8 (1)
    public static final int OFF_FOOT_INDEX = 85;         // uint8 (1)
    public static final int OFF_FORWARD_INDEX = 86;      // uint8 (1)
    public static final int OFF_LATERAL_INDEX = 87;      // uint8 (1)
    public static final int OFF_VOCAB_SIZE = 88;         // uint32 (4)
    public static final int OFF_STRIDE_0 = 92;           // uint8 (1)
    public static final int OFF_STRIDE_1 = 93;           // uint8 (1)
    public static final int OFF_STRIDE_2 = 94;           // uint8 (1)
    public static final int OFF_RESERVED0 = 95;          // uint8 (1)

    // Block Channels per LOD (34 bytes per cell * 4096 = 139,264 bytes per LOD):
    // - uint16 blockId (8192 bytes)
    // - float32 occupancy (16384 bytes)
    // - float32 passability (16384 bytes)
    // - float32 hazard (16384 bytes)
    // - float32 hardness (16384 bytes)
    // - float32 lightLevel (16384 bytes)
    // - float32 fluidFlowX (16384 bytes)
    // - float32 fluidFlowZ (16384 bytes)
    // - float32 visibility (16384 bytes)
    public static final int OFF_BLOCK_ID = 0;
    public static final int OFF_OCCUPANCY = 8192;
    public static final int OFF_PASSABILITY = 24576;
    public static final int OFF_HAZARD = 40960;
    public static final int OFF_HARDNESS = 57344;
    public static final int OFF_LIGHT = 73728;
    public static final int OFF_FLUID_X = 90112;
    public static final int OFF_FLUID_Z = 106496;
    public static final int OFF_VISIBILITY = 122880;

    public static final int BLOCK_BYTES_PER_LOD = CELLS_PER_LOD * 34; // 139,264
    public static final int TOTAL_BLOCK_BYTES = BLOCK_BYTES_PER_LOD * NUM_LODS; // 417,792

    // Entity Channels per LOD (10 bytes per cell * 4096 = 40,960 bytes per LOD):
    // - uint8 entityCount (4096 bytes)
    // - uint8 entityTypeId (4096 bytes)
    // - float32 isHostile (16384 bytes)
    // - float32 totalHealth (16384 bytes)
    public static final int OFF_ENTITY_COUNT = 0;
    public static final int OFF_ENTITY_TYPE_ID = 4096;
    public static final int OFF_IS_HOSTILE = 8192;
    public static final int OFF_TOTAL_HEALTH = 24576;

    public static final int ENTITY_BYTES_PER_LOD = CELLS_PER_LOD * 10; // 40,960
    public static final int TOTAL_ENTITY_BYTES = ENTITY_BYTES_PER_LOD * NUM_LODS; // 122,880

    // Auxiliary / Crosshair Targeting & Proprioception
    public static final int AUX_BYTES = 256;

    // Slot Size (padded to 4 KB page alignment)
    // 417,792 + 122,880 + 256 = 540,928 -> rounded up to 544,768 (133 * 4096)
    public static final int UNPADDED_SLOT_SIZE = TOTAL_BLOCK_BYTES + TOTAL_ENTITY_BYTES + AUX_BYTES;
    public static final int SLOT_SIZE = ((UNPADDED_SLOT_SIZE + 4095) / 4096) * 4096; // 544,768

    // Slot Offsets
    public static final int SLOT_0_OFFSET = HEADER_SIZE; // 4096
    public static final int SLOT_1_OFFSET = HEADER_SIZE + SLOT_SIZE; // 4096 + 544768 = 548,864

    // Total MMF File Size
    public static final int TOTAL_FILE_SIZE = HEADER_SIZE + SLOT_SIZE * 2; // 1,093,632 bytes (~1.04 MB)

    public static int getSlotOffset(int slotIndex) {
        return HEADER_SIZE + (slotIndex & 1) * SLOT_SIZE;
    }

    public static int getLodBlockOffset(int slotOffset, int lod) {
        return slotOffset + lod * BLOCK_BYTES_PER_LOD;
    }

    public static int getLodEntityOffset(int slotOffset, int lod) {
        return slotOffset + TOTAL_BLOCK_BYTES + lod * ENTITY_BYTES_PER_LOD;
    }

    public static int getAuxOffset(int slotOffset) {
        return slotOffset + TOTAL_BLOCK_BYTES + TOTAL_ENTITY_BYTES;
    }
}

