package org.sqsense.squared_sense_mc.client.observation.sampler;

import com.google.gson.JsonArray;
import java.util.Map;

/**
 * Result of a 16x16x16 sampling pass.
 */
public record SamplingResult(
    JsonArray gridArray,
    Map<String, Integer> globalCounts,
    int totalCells,
    int airCells,
    int nonAirCells
) {}

