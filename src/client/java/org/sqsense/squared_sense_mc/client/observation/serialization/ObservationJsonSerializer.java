package org.sqsense.squared_sense_mc.client.observation.serialization;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.util.math.BlockPos;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;
import org.sqsense.squared_sense_mc.client.observation.sampler.SamplingResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

/**
 * Handles JSON construction and off-thread disk serialization for observation data.
 */
public class ObservationJsonSerializer {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static JsonObject buildSingleLodJson(
        ClientWorld world,
        BlockPos playerPos,
        int lod,
        ObservationTransform transform,
        SamplingResult result
    ) {
        JsonObject root = new JsonObject();
        JsonObject meta = buildCommonMetadata(world, playerPos, transform);
        meta.addProperty("lod", lod);
        meta.addProperty("stride", transform.getStride());

        int span = ObservationConfig.GRID_SIZE * transform.getStride();
        JsonObject coverage = new JsonObject();
        coverage.addProperty("sizeX", span);
        coverage.addProperty("sizeY", span);
        coverage.addProperty("sizeZ", span);
        meta.add("worldCoverage", coverage);

        JsonObject stats = buildStatsJson(result, transform.getStride());
        meta.add("stats", stats);

        root.add("metadata", meta);
        root.add("grid", result.gridArray());
        return root;
    }

    public static JsonObject buildAllLodsJson(
        ClientWorld world,
        BlockPos playerPos,
        ObservationTransform transform,
        SamplingResult res0,
        SamplingResult res1,
        SamplingResult res2
    ) {
        JsonObject root = new JsonObject();
        JsonObject meta = buildCommonMetadata(world, playerPos, transform);

        root.add("metadata", meta);
        root.add("b0_lod0", res0.gridArray());
        root.add("b1_lod1", res1.gridArray());
        root.add("b2_lod2", res2.gridArray());
        return root;
    }

    private static JsonObject buildCommonMetadata(
        ClientWorld world,
        BlockPos playerPos,
        ObservationTransform transform
    ) {
        JsonObject meta = new JsonObject();
        meta.addProperty("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
        meta.addProperty("dimension", world.getRegistryKey().getValue().toString());
        meta.addProperty("gridSize", ObservationConfig.GRID_SIZE);
        meta.addProperty("coordinateFrame", "relative_player_feet_egocentric");
        meta.addProperty("downsampling", "priority_hierarchy");
        meta.addProperty("vocabSize", BlockVocabulary.getVocabSize());

        com.google.gson.JsonArray channels = new com.google.gson.JsonArray();
        channels.add("id");
        channels.add("name");
        channels.add("dx");
        channels.add("dy");
        channels.add("dz");
        channels.add("occupancy");
        channels.add("passability");
        channels.add("hazard");
        channels.add("hardness");
        channels.add("lightLevel");
        channels.add("fluidFlow");
        meta.add("channels", channels);

        JsonObject camera = new JsonObject();
        camera.addProperty("rawYaw", transform.getRawYaw());
        camera.addProperty("rawPitch", transform.getRawPitch());
        camera.addProperty("snappedYaw", transform.getCardinal().getYawDegrees());
        camera.addProperty("cardinalDirection", transform.getCardinal().name());
        camera.addProperty("footIndex", transform.getFootIndex());
        camera.addProperty("playerForwardIndex", ObservationConfig.PLAYER_FORWARD_INDEX);
        camera.addProperty("playerLateralIndex", ObservationConfig.PLAYER_LATERAL_INDEX);
        meta.add("camera", camera);

        JsonObject playerObj = new JsonObject();
        playerObj.addProperty("x", playerPos.getX());
        playerObj.addProperty("y", playerPos.getY());
        playerObj.addProperty("z", playerPos.getZ());
        meta.add("playerWorldPos", playerObj);

        return meta;
    }

    private static JsonObject buildStatsJson(SamplingResult result, int stride) {
        JsonObject stats = new JsonObject();
        stats.addProperty("totalCells", result.totalCells());
        stats.addProperty("totalWorldBlocks", result.totalCells() * stride * stride * stride);
        stats.addProperty("airCells", result.airCells());
        stats.addProperty("nonAirCells", result.nonAirCells());

        JsonObject countsObj = new JsonObject();
        result.globalCounts().entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .forEach(entry -> countsObj.addProperty(entry.getKey(), entry.getValue()));
        stats.add("blockCounts", countsObj);
        return stats;
    }

    public static CompletableFuture<Void> writeAsync(
        JsonObject json,
        Path targetFile,
        Path latestFile,
        Path latestLodFile,
        Consumer<Path> onSuccess,
        Consumer<Exception> onError
    ) {
        return CompletableFuture.runAsync(() -> {
            try {
                if (targetFile.getParent() != null) {
                    Files.createDirectories(targetFile.getParent());
                    Path vocabFile = targetFile.getParent().resolve("block_vocab.json");
                    if (!Files.exists(vocabFile)) {
                        BlockVocabulary.exportVocabulary(vocabFile);
                    }
                }
                String jsonString = GSON.toJson(json);
                Files.writeString(targetFile, jsonString, StandardCharsets.UTF_8);
                if (latestFile != null) {
                    Files.writeString(latestFile, jsonString, StandardCharsets.UTF_8);
                }
                if (latestLodFile != null) {
                    Files.writeString(latestLodFile, jsonString, StandardCharsets.UTF_8);
                }
                onSuccess.accept(targetFile);
            } catch (IOException e) {
                onError.accept(e);
            }
        });
    }
}

