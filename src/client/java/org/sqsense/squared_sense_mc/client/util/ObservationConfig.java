package org.sqsense.squared_sense_mc.client.util;

public final class ObservationConfig {
    private ObservationConfig() {}

    /**
     * Fixed dimension for each axis of the observation tensor (16x16x16).
     */
    public static final int GRID_SIZE = 16;

    /**
     * Strides in real-world 1-meter blocks for each LOD level.
     * LOD 0 -> Stride 1 (spans 16x16x16 world volume)
     * LOD 1 -> Stride 2 (spans 32x32x32 world volume)
     * LOD 2 -> Stride 4 (spans 64x64x64 world volume)
     */
    public static final int[] LOD_STRIDES = {1, 2, 4};

    public static int getStride(int lod) {
        if (lod < 0 || lod >= LOD_STRIDES.length) {
            throw new IllegalArgumentException("Unsupported LOD index: " + lod + " (must be 0, 1, or 2)");
        }
        return LOD_STRIDES[lod];
    }
}

