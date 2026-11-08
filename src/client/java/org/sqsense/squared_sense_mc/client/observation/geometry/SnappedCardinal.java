package org.sqsense.squared_sense_mc.client.observation.geometry;

/**
 * The 4 snapped cardinal directions with their respective unit vectors
 * in Minecraft's horizontal coordinate frame (+X East, -X West, +Z South, -Z North).
 */
public enum SnappedCardinal {
    SOUTH(0, 0, 1, -1, 0),
    WEST(90, -1, 0, 0, -1),
    NORTH(180, 0, -1, 1, 0),
    EAST(270, 1, 0, 0, 1);

    private final int yawDegrees;
    private final int forwardX;
    private final int forwardZ;
    private final int rightX;
    private final int rightZ;

    SnappedCardinal(int yawDegrees, int forwardX, int forwardZ, int rightX, int rightZ) {
        this.yawDegrees = yawDegrees;
        this.forwardX = forwardX;
        this.forwardZ = forwardZ;
        this.rightX = rightX;
        this.rightZ = rightZ;
    }

    public int getYawDegrees() {
        return yawDegrees;
    }

    public int getForwardX() {
        return forwardX;
    }

    public int getForwardZ() {
        return forwardZ;
    }

    public int getRightX() {
        return rightX;
    }

    public int getRightZ() {
        return rightZ;
    }

    /**
     * Transforms local lateral (x) and forward (z) offsets into world delta coordinates (dx, dz).
     */
    public int toWorldDeltaX(int xLocal, int zLocal) {
        return xLocal * rightX + zLocal * forwardX;
    }

    /**
     * Transforms local lateral (x) and forward (z) offsets into world delta coordinates (dx, dz).
     */
    public int toWorldDeltaZ(int xLocal, int zLocal) {
        return xLocal * rightZ + zLocal * forwardZ;
    }

    /**
     * Transforms world delta horizontal coordinates (dx, dz) into local lateral offset (xLocal).
     */
    public int toLocalX(int deltaX, int deltaZ) {
        return deltaX * rightX + deltaZ * rightZ;
    }

    /**
     * Transforms world delta horizontal coordinates (dx, dz) into local forward offset (zLocal).
     */
    public int toLocalZ(int deltaX, int deltaZ) {
        return deltaX * forwardX + deltaZ * forwardZ;
    }

    /**
     * Continuous floating-point variant for precise entity offsets.
     */
    public double toLocalX(double deltaX, double deltaZ) {
        return deltaX * rightX + deltaZ * rightZ;
    }

    /**
     * Continuous floating-point variant for precise entity offsets.
     */
    public double toLocalZ(double deltaX, double deltaZ) {
        return deltaX * forwardX + deltaZ * forwardZ;
    }
}

