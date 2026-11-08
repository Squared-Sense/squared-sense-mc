package org.sqsense.squared_sense_mc.client.observation.entity;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Samples entities within the matching observation bounding box and projects them into
 * a fixed 16x16x16 entity tensor grid (Task 10):
 * - Channels per cell: entityCount, isHostile (max-pooled), totalHealth.
 * - Filters out the observing player and dead/removed entities.
 */
public class EntityGridSampler {

    public static EntitySamplingResult sample(
        ClientWorld world,
        ClientPlayerEntity player,
        BlockPos playerPos,
        ObservationTransform transform
    ) {
        final int size = ObservationConfig.GRID_SIZE;
        final int totalCells = size * size * size;

        int[] count = new int[totalCells];
        float[] isHostile = new float[totalCells];
        float[] totalHealth = new float[totalCells];
        @SuppressWarnings("unchecked")
        List<EntityEntry>[] entries = new List[totalCells];

        Box queryBox = transform.getWorldBoundingBox(playerPos);
        List<Entity> entities = world.getEntitiesByClass(
            Entity.class,
            queryBox,
            e -> e != player && e.isAlive() && !e.isRemoved()
        );

        int totalEntities = 0;
        int hostileCount = 0;
        int passiveCount = 0;
        Map<String, Integer> countsByType = new LinkedHashMap<>();

        int playerX = playerPos.getX();
        int playerY = playerPos.getY();
        int playerZ = playerPos.getZ();

        for (Entity entity : entities) {
            int ebx = entity.getBlockX();
            int eby = entity.getBlockY();
            int ebz = entity.getBlockZ();

            int wdx = ebx - playerX;
            int wdy = eby - playerY;
            int wdz = ebz - playerZ;

            int gx = transform.getGridX(wdx, wdz);
            int gy = transform.getGridY(wdy);
            int gz = transform.getGridZ(wdx, wdz);

            if (!transform.isInGrid(gx, gy, gz)) {
                continue;
            }

            int index = (gx * size + gy) * size + gz;
            boolean hostile = EntityClassifier.isHostile(entity);
            float health = EntityClassifier.getHealth(entity);
            String typeId = EntityClassifier.getTypeId(entity);

            count[index]++;
            if (hostile) {
                isHostile[index] = 1.0f;
            }
            totalHealth[index] += health;

            double continuousWdx = entity.getX() - playerX;
            double continuousWdy = entity.getY() - playerY;
            double continuousWdz = entity.getZ() - playerZ;

            double localX = transform.getCardinal().toLocalX(continuousWdx, continuousWdz);
            double localZ = transform.getCardinal().toLocalZ(continuousWdx, continuousWdz);
            double localY = continuousWdy;

            if (entries[index] == null) {
                entries[index] = new ArrayList<>(2);
            }
            entries[index].add(new EntityEntry(typeId, health, hostile, localX, localY, localZ));

            totalEntities++;
            if (hostile) {
                hostileCount++;
            } else {
                passiveCount++;
            }
            countsByType.put(typeId, countsByType.getOrDefault(typeId, 0) + 1);
        }

        JsonArray xArray = new JsonArray();
        for (int gx = 0; gx < size; gx++) {
            JsonArray yArray = new JsonArray();
            for (int gy = 0; gy < size; gy++) {
                int dy = transform.getDeltaY(gy);
                JsonArray zArray = new JsonArray();
                for (int gz = 0; gz < size; gz++) {
                    int dx = transform.getDeltaX(gx, gz);
                    int dz = transform.getDeltaZ(gx, gz);
                    int index = (gx * size + gy) * size + gz;

                    JsonObject cellObj = new JsonObject();
                    cellObj.addProperty("dx", dx);
                    cellObj.addProperty("dy", dy);
                    cellObj.addProperty("dz", dz);

                    int c = count[index];
                    cellObj.addProperty("entityCount", c);
                    cellObj.addProperty("isHostile", isHostile[index]);
                    cellObj.addProperty("totalHealth", Math.round(totalHealth[index] * 100.0f) / 100.0f);

                    if (c > 0 && entries[index] != null) {
                        JsonArray entList = new JsonArray();
                        for (EntityEntry entry : entries[index]) {
                            entList.add(entry.toJsonObject());
                        }
                        cellObj.add("entities", entList);
                    }
                    zArray.add(cellObj);
                }
                yArray.add(zArray);
            }
            xArray.add(yArray);
        }

        return new EntitySamplingResult(xArray, totalEntities, hostileCount, passiveCount, countsByType);
    }
}

