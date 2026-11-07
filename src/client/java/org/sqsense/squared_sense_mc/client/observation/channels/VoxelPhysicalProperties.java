package org.sqsense.squared_sense_mc.client.observation.channels;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

/**
 * Multi-channel aggregated physical properties for a voxel cell (Task 8):
 * - occupancy: solid density [0.0, 1.0]
 * - passability: movement permeability [0.0, 1.0] (1.0 air, 0.5 water/cobweb, 0.0 solid)
 * - hazard: binary flag (1.0 if ANY sub-block is lava/fire/hazard, else 0.0)
 * - hardness: average block break hardness in cell
 * - lightLevel: combined block + sky light normalized to [0.0, 1.0]
 * - flowX, flowZ: fluid velocity vector rotated into egocentric frame [-1.0, 1.0]
 */
public record VoxelPhysicalProperties(
    float occupancy,
    float passability,
    float hazard,
    float hardness,
    float lightLevel,
    float flowX,
    float flowZ
) {
    public JsonObject toJsonObject() {
        JsonObject obj = new JsonObject();
        obj.addProperty("occupancy", round(occupancy));
        obj.addProperty("passability", round(passability));
        obj.addProperty("hazard", hazard);
        obj.addProperty("hardness", round(hardness));
        obj.addProperty("lightLevel", round(lightLevel));

        JsonArray flow = new JsonArray();
        flow.add(round(flowX));
        flow.add(round(flowZ));
        obj.add("fluidFlow", flow);

        return obj;
    }

    private static float round(float val) {
        return Math.round(val * 1000.0f) / 1000.0f;
    }
}

