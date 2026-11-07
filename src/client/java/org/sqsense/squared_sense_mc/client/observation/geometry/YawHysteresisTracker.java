package org.sqsense.squared_sense_mc.client.observation.geometry;

/**
 * Stateful tracker that snaps continuous yaw to cardinal directions (0, 90, 180, 270 degrees)
 * with a +/-10 degree hysteresis deadband around the 45 degree diagonal boundaries
 * to prevent grid-thrashing.
 */
public class YawHysteresisTracker {
    private SnappedCardinal current = null;

    /**
     * Updates the tracker with the player's current yaw and returns the snapped cardinal direction.
     *
     * @param rawYaw Continuous yaw angle in degrees (unbounded).
     * @return Snapped cardinal direction.
     */
    public SnappedCardinal update(float rawYaw) {
        float yaw = normalizeYaw(rawYaw);

        if (current == null) {
            current = getNearest(yaw);
            return current;
        }

        current = switch (current) {
            case SOUTH -> evaluateFromSouth(yaw);
            case WEST -> evaluateFromWest(yaw);
            case NORTH -> evaluateFromNorth(yaw);
            case EAST -> evaluateFromEast(yaw);
        };

        return current;
    }

    public SnappedCardinal getCurrent() {
        return current != null ? current : SnappedCardinal.SOUTH;
    }

    public void reset() {
        current = null;
    }

    public static float normalizeYaw(float yaw) {
        return (yaw % 360.0f + 360.0f) % 360.0f;
    }

    public static SnappedCardinal getNearest(float normalizedYaw) {
        if (normalizedYaw >= 45.0f && normalizedYaw < 135.0f) {
            return SnappedCardinal.WEST;
        } else if (normalizedYaw >= 135.0f && normalizedYaw < 225.0f) {
            return SnappedCardinal.NORTH;
        } else if (normalizedYaw >= 225.0f && normalizedYaw < 315.0f) {
            return SnappedCardinal.EAST;
        } else {
            return SnappedCardinal.SOUTH;
        }
    }

    private SnappedCardinal evaluateFromSouth(float yaw) {
        if (yaw >= 55.0f && yaw < 145.0f) {
            return SnappedCardinal.WEST;
        } else if (yaw >= 145.0f && yaw < 235.0f) {
            return SnappedCardinal.NORTH;
        } else if (yaw >= 235.0f && yaw <= 305.0f) {
            return SnappedCardinal.EAST;
        }
        return SnappedCardinal.SOUTH;
    }

    private SnappedCardinal evaluateFromWest(float yaw) {
        if (yaw < 35.0f || yaw > 325.0f) {
            return SnappedCardinal.SOUTH;
        } else if (yaw >= 145.0f && yaw < 235.0f) {
            return SnappedCardinal.NORTH;
        } else if (yaw >= 235.0f && yaw <= 325.0f) {
            return SnappedCardinal.EAST;
        }
        return SnappedCardinal.WEST;
    }

    private SnappedCardinal evaluateFromNorth(float yaw) {
        if (yaw >= 235.0f && yaw < 325.0f) {
            return SnappedCardinal.EAST;
        } else if (yaw <= 125.0f && yaw >= 35.0f) {
            return SnappedCardinal.WEST;
        } else if (yaw < 35.0f || yaw >= 325.0f) {
            return SnappedCardinal.SOUTH;
        }
        return SnappedCardinal.NORTH;
    }

    private SnappedCardinal evaluateFromEast(float yaw) {
        if (yaw < 215.0f && yaw >= 125.0f) {
            return SnappedCardinal.NORTH;
        } else if (yaw < 125.0f && yaw >= 35.0f) {
            return SnappedCardinal.WEST;
        } else if (yaw > 325.0f || yaw < 35.0f) {
            return SnappedCardinal.SOUTH;
        }
        return SnappedCardinal.EAST;
    }
}

