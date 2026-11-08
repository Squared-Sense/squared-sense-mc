package org.sqsense.squared_sense_mc.client.observation.entity;

import com.google.gson.JsonArray;
import java.util.Map;

/**
 * Encapsulates the sampled entity tensor grid (En) and aggregate entity statistics (Task 10).
 */
public record EntitySamplingResult(
    JsonArray gridArray,
    int totalEntities,
    int hostileEntities,
    int passiveEntities,
    Map<String, Integer> countsByType
) {}

