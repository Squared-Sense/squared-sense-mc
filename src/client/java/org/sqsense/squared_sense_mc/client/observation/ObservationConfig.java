package org.sqsense.squared_sense_mc.client.observation;

public final class ObservationConfig {
    private ObservationConfig() {}

    /**
     * Fixed dimension for each axis of the observation tensor (16x16x16).
     */
    public static final int GRID_SIZE = 16;

    /**
     * Grid index along the forward axis (gz) where the player sits (index 2: 2 behind, 13 in front).
     */
    public static final int PLAYER_FORWARD_INDEX = 2;

    /**
     * Grid index along the lateral axis (gx) where the player sits (symmetric [-8, +7]).
     */
    public static final int PLAYER_LATERAL_INDEX = 8;

    /**
     * Default foot index along vertical axis (gy) when pitch is 0 degrees (looking horizontal).
     */
    public static final int DEFAULT_FOOT_INDEX = 6;

    /**
     * Maximum foot index along vertical axis (gy) when pitch is +90 degrees (looking down).
     * Exposes 12 blocks below the player's feet.
     */
    public static final int MAX_FOOT_INDEX_LOOKING_DOWN = 12;

    /**
     * Minimum foot index along vertical axis (gy) when pitch is -90 degrees (looking up).
     * Exposes 14 blocks above the player's feet.
     */
    public static final int MIN_FOOT_INDEX_LOOKING_UP = 1;

    /**
     * Deadband half-width in degrees around the 45-degree diagonal boundaries.
     */
    public static final float YAW_DEADBAND_DEGREES = 10.0f;

    /**
     * Strides in real-world 1-meter blocks for each LOD level.
     * LOD 0 -> Stride 1 (spans 16x16x16 world volume)
     * LOD 1 -> Stride 2 (spans 32x32x32 world volume)
     * LOD 2 -> Stride 4 (spans 64x64x64 world volume)
     */
    public static final int[] LOD_STRIDES = {1, 2, 4};

    /**
     * Default streaming cadence in Minecraft client game ticks (2 ticks = 10 Hz / 100 ms).
     */
    public static final int DEFAULT_STREAM_INTERVAL_TICKS = 2;

    public static int getStride(int lod) {
        if (lod < 0 || lod >= LOD_STRIDES.length) {
            throw new IllegalArgumentException("Unsupported LOD index: " + lod + " (must be 0, 1, or 2)");
        }
        return LOD_STRIDES[lod];
    }

    /**
     * Phase 5 Visibility & Occlusion Masking toggle.
     * When true, blocks completely surrounded by opaque blocks are culled to MASKED_ID (1) with visibility 0.0.
     * When false, all blocks retain their raw omniscient block IDs with visibility 1.0.
     */
    public static volatile boolean visibilityFilterEnabled = true;

    public static boolean toggleVisibilityFilter() {
        visibilityFilterEnabled = !visibilityFilterEnabled;
        return visibilityFilterEnabled;
    }
}

