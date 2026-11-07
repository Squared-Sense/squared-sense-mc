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
}

