package org.sqsense.squared_sense_mc.client.observation.geometry;

import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;

/**
 * Calculates the vertical window foot index (gy where player's feet sit) based on player head pitch.
 * In Minecraft:
 *  - Pitch +90: Looking straight DOWN. Foot level sits at index 12 (exposing 12 blocks below).
 *  - Pitch   0: Looking horizontal. Foot level sits at index 6.
 *  - Pitch -90: Looking straight UP. Foot level sits at index 1 (exposing 14 blocks above).
 */
public final class PitchWindowCalculator {
    private PitchWindowCalculator() {}

    public static int calculateFootIndex(float pitch) {
        float clampedPitch = Math.max(-90.0f, Math.min(90.0f, pitch));

        if (clampedPitch >= 0) {
            // Pitch 0 -> 90 maps to footIndex 6 -> 12
            float factor = clampedPitch / 90.0f;
            return Math.round(ObservationConfig.DEFAULT_FOOT_INDEX
                + factor * (ObservationConfig.MAX_FOOT_INDEX_LOOKING_DOWN - ObservationConfig.DEFAULT_FOOT_INDEX));
        } else {
            // Pitch 0 -> -90 maps to footIndex 6 -> 1
            float factor = clampedPitch / -90.0f;
            return Math.round(ObservationConfig.DEFAULT_FOOT_INDEX
                - factor * (ObservationConfig.DEFAULT_FOOT_INDEX - ObservationConfig.MIN_FOOT_INDEX_LOOKING_UP));
        }
    }
}

