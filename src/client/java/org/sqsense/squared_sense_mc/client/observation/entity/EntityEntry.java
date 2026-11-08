package org.sqsense.squared_sense_mc.client.observation.entity;

import com.google.gson.JsonObject;

/**
 * Details of an individual entity mapped within a voxel cell (Task 10).
 */
public record EntityEntry(
    String type,
    float health,
    boolean hostile,
    double localX,
    double localY,
    double localZ
) {
    public JsonObject toJsonObject() {
        JsonObject obj = new JsonObject();
        obj.addProperty("type", type);
        obj.addProperty("health", Math.round(health * 100.0f) / 100.0f);
        obj.addProperty("hostile", hostile);
        obj.addProperty("localX", Math.round(localX * 100.0) / 100.0);
        obj.addProperty("localY", Math.round(localY * 100.0) / 100.0);
        obj.addProperty("localZ", Math.round(localZ * 100.0) / 100.0);
        return obj;
    }
}

