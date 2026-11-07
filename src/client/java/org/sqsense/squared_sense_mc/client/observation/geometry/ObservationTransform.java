package org.sqsense.squared_sense_mc.client.observation.geometry;

import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;

/**
 * Encapsulates the spatial frame transformation for an observation capture:
 * - Rotates into egocentric space (+Z is forward, +X is right, +Y is up).
 * - Applies asymmetric forward offset (player at gz=2).
 * - Applies symmetric lateral offset (gx in [-8, +7]).
 * - Applies pitch-based vertical translation (foot at gy=footIndex).
 * - Scales by LOD stride in 1-meter block units.
 */
public class ObservationTransform {
    private final float rawYaw;
    private final float rawPitch;
    private final SnappedCardinal cardinal;
    private final int footIndex;
    private final int stride;

    private final int[] xLocalOffsets;
    private final int[] yLocalOffsets;
    private final int[] zLocalOffsets;

    public ObservationTransform(float rawYaw, float rawPitch, SnappedCardinal cardinal, int stride) {
        this.rawYaw = rawYaw;
        this.rawPitch = rawPitch;
        this.cardinal = cardinal;
        this.stride = stride;
        this.footIndex = PitchWindowCalculator.calculateFootIndex(rawPitch);

        final int size = ObservationConfig.GRID_SIZE;
        this.xLocalOffsets = new int[size];
        this.yLocalOffsets = new int[size];
        this.zLocalOffsets = new int[size];

        for (int i = 0; i < size; i++) {
            // Lateral: symmetric [-8, +7] relative to player
            this.xLocalOffsets[i] = (i - ObservationConfig.PLAYER_LATERAL_INDEX) * stride;
            // Vertical: relative to player's feet at footIndex
            this.yLocalOffsets[i] = (i - this.footIndex) * stride;
            // Forward: player sits at index 2 (2 behind, 13 in front)
            this.zLocalOffsets[i] = (i - ObservationConfig.PLAYER_FORWARD_INDEX) * stride;
        }
    }

    public int getDeltaX(int gx, int gz) {
        return cardinal.toWorldDeltaX(xLocalOffsets[gx], zLocalOffsets[gz]);
    }

    public int getDeltaY(int gy) {
        return yLocalOffsets[gy];
    }

    public int getDeltaZ(int gx, int gz) {
        return cardinal.toWorldDeltaZ(xLocalOffsets[gx], zLocalOffsets[gz]);
    }

    public float getRawYaw() {
        return rawYaw;
    }

    public float getRawPitch() {
        return rawPitch;
    }

    public SnappedCardinal getCardinal() {
        return cardinal;
    }

    public int getFootIndex() {
        return footIndex;
    }

    public int getStride() {
        return stride;
    }

    public int getXLocalOffset(int gx) {
        return xLocalOffsets[gx];
    }

    public int getYLocalOffset(int gy) {
        return yLocalOffsets[gy];
    }

    public int getZLocalOffset(int gz) {
        return zLocalOffsets[gz];
    }
}

