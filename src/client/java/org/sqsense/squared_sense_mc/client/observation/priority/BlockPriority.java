package org.sqsense.squared_sense_mc.client.observation.priority;

/**
 * Priority tiers for intelligent downsampling (Task 7):
 * Higher rank blocks are preserved even when in the minority within a downsampled sub-volume.
 *
 * Rank Hierarchy:
 * HAZARD (6) > INTERACTIVE (5) > LIQUID (4) > FOLIAGE (3) > SOLID (2) > AIR (1)
 */
public enum BlockPriority {
    HAZARD(6, "hazard"),
    INTERACTIVE(5, "interactive"),
    LIQUID(4, "liquid"),
    FOLIAGE(3, "foliage"),
    SOLID(2, "solid"),
    AIR(1, "air");

    private final int rank;
    private final String label;

    BlockPriority(int rank, String label) {
        this.rank = rank;
        this.label = label;
    }

    public int getRank() {
        return rank;
    }

    public String getLabel() {
        return label;
    }

    public boolean isHigherPriorityThan(BlockPriority other) {
        return this.rank > other.rank;
    }
}

