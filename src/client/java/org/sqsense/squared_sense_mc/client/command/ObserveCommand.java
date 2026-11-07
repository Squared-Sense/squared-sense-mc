package org.sqsense.squared_sense_mc.client.command;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.brigadier.Command;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.registry.Registries;
import net.minecraft.text.ClickEvent;
import net.minecraft.text.MutableText;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sqsense.squared_sense_mc.client.util.ObservationConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * High-performance observational retrieval command for RL training.
 * Samples fixed 16x16x16 grid tensors in relative coordinates (dx, dy, dz)
 * anchored to the player's feet across LODs 0, 1, and 2 (strides 1, 2, and 4).
 */
public class ObserveCommand {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    // Fast-path Block -> String identifier cache to avoid registry overhead during sampling
    private static final Map<Block, String> BLOCK_ID_CACHE = new ConcurrentHashMap<>();

    public static void register(CommandDispatcher<FabricClientCommandSource> dispatcher) {
        var filenameArg = ClientCommandManager.argument("filename", StringArgumentType.string())
            .executes(ctx -> executeSingle(
                ctx,
                IntegerArgumentType.getInteger(ctx, "lod"),
                StringArgumentType.getString(ctx, "filename")
            ));

        var lodArg = ClientCommandManager.argument("lod", IntegerArgumentType.integer(0, 2))
            .executes(ctx -> executeSingle(
                ctx,
                IntegerArgumentType.getInteger(ctx, "lod"),
                null
            ))
            .then(filenameArg);

        var allFilenameArg = ClientCommandManager.argument("filename", StringArgumentType.string())
            .executes(ctx -> executeAll(
                ctx,
                StringArgumentType.getString(ctx, "filename")
            ));

        var allLiteral = ClientCommandManager.literal("all")
            .executes(ctx -> executeAll(ctx, null))
            .then(allFilenameArg);

        var observeNode = dispatcher.register(
            ClientCommandManager.literal("observe")
                .executes(ctx -> executeSingle(ctx, 0, null))
                .then(allLiteral)
                .then(lodArg)
        );

        // Alias: /obs -> /observe
        dispatcher.register(ClientCommandManager.literal("obs").redirect(observeNode));
    }

    private static int executeSingle(
        CommandContext<FabricClientCommandSource> context,
        int lod,
        String customFilename
    ) {
        var source = context.getSource();
        MinecraftClient client = source.getClient();
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;

        if (client == null || player == null || world == null) {
            source.sendError(Text.literal("§c[Observe] World or player not available."));
            return 0;
        }

        BlockPos playerPos = player.getBlockPos();
        int stride = ObservationConfig.getStride(lod);

        // Precompute relative 1-meter offsets: indices [-8 .. +7] * stride
        int[] relCoords = precomputeRelativeOffsets(stride);

        // Sample the 16x16x16 grid tensor
        SamplingResult result = sampleGrid(world, playerPos, stride, relCoords);

        // Build JSON output
        JsonObject root = new JsonObject();
        JsonObject meta = buildMetadata(world, playerPos, lod, stride, relCoords, result);
        root.add("metadata", meta);
        root.add("grid", result.gridArray);

        // Determine destination file
        Path baseDir = client.runDirectory.toPath().resolve("observations");
        String fileName;
        if (customFilename != null && !customFilename.isBlank()) {
            String sanitized = customFilename.trim().replaceAll("[\\\\/:*?\"<>|]", "_");
            fileName = sanitized.endsWith(".json") ? sanitized : sanitized + ".json";
        } else {
            String timestamp = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss").format(LocalDateTime.now());
            fileName = String.format("obs_lod%d_%s.json", lod, timestamp);
        }

        Path targetFile = baseDir.resolve(fileName).normalize();
        Path latestFile = baseDir.resolve("obs_latest.json");
        Path latestLodFile = baseDir.resolve(String.format("obs_lod%d_latest.json", lod));

        // Asynchronous file write to prevent frame drops in game loop
        CompletableFuture.runAsync(() -> {
            try {
                Files.createDirectories(baseDir);
                String jsonString = GSON.toJson(root);
                Files.writeString(targetFile, jsonString, StandardCharsets.UTF_8);
                Files.writeString(latestFile, jsonString, StandardCharsets.UTF_8);
                Files.writeString(latestLodFile, jsonString, StandardCharsets.UTF_8);

                LOGGER.info("Observe: Captured LOD {} to {}", lod, targetFile.toAbsolutePath());

                client.execute(() -> {
                    if (client.player != null) {
                        int coverage = ObservationConfig.GRID_SIZE * stride;
                        MutableText msg = Text.literal("§a[Observe]§r Captured LOD §b" + lod + "§r ("
                            + ObservationConfig.GRID_SIZE + "x" + ObservationConfig.GRID_SIZE + "x" + ObservationConfig.GRID_SIZE
                            + " grid, stride " + stride + ", spans §e" + coverage + "x" + coverage + "x" + coverage + "m§r) to ")
                            .append(Text.literal("§6" + targetFile.getFileName() + "§r").styled(style -> style
                                .withUnderline(true)
                                .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, targetFile.toAbsolutePath().toString()))
                            ))
                            .append(Text.literal(" §7[Click to open]§r"));

                        source.sendFeedback(msg);
                    }
                });
            } catch (IOException e) {
                LOGGER.error("Observe: Failed to save file", e);
                client.execute(() -> {
                    source.sendError(Text.literal("§c[Observe] Failed to save file: " + e.getMessage()));
                });
            }
        });

        return Command.SINGLE_SUCCESS;
    }

    private static int executeAll(
        CommandContext<FabricClientCommandSource> context,
        String customFilename
    ) {
        var source = context.getSource();
        MinecraftClient client = source.getClient();
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;

        if (client == null || player == null || world == null) {
            source.sendError(Text.literal("§c[Observe] World or player not available."));
            return 0;
        }

        BlockPos playerPos = player.getBlockPos();

        // Sample all 3 LOD tensors: B0, B1, B2
        SamplingResult result0 = sampleGrid(world, playerPos, ObservationConfig.getStride(0), precomputeRelativeOffsets(ObservationConfig.getStride(0)));
        SamplingResult result1 = sampleGrid(world, playerPos, ObservationConfig.getStride(1), precomputeRelativeOffsets(ObservationConfig.getStride(1)));
        SamplingResult result2 = sampleGrid(world, playerPos, ObservationConfig.getStride(2), precomputeRelativeOffsets(ObservationConfig.getStride(2)));

        JsonObject root = new JsonObject();

        JsonObject meta = new JsonObject();
        meta.addProperty("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
        meta.addProperty("dimension", world.getRegistryKey().getValue().toString());
        meta.addProperty("gridSize", ObservationConfig.GRID_SIZE);
        meta.addProperty("coordinateFrame", "relative_player_feet");

        JsonObject playerObj = new JsonObject();
        playerObj.addProperty("x", playerPos.getX());
        playerObj.addProperty("y", playerPos.getY());
        playerObj.addProperty("z", playerPos.getZ());
        meta.add("playerWorldPos", playerObj);

        root.add("metadata", meta);
        root.add("b0", result0.gridArray);
        root.add("b1", result1.gridArray);
        root.add("b2", result2.gridArray);

        Path baseDir = client.runDirectory.toPath().resolve("observations");
        String fileName;
        if (customFilename != null && !customFilename.isBlank()) {
            String sanitized = customFilename.trim().replaceAll("[\\\\/:*?\"<>|]", "_");
            fileName = sanitized.endsWith(".json") ? sanitized : sanitized + ".json";
        } else {
            String timestamp = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss").format(LocalDateTime.now());
            fileName = String.format("obs_all_%s.json", timestamp);
        }

        Path targetFile = baseDir.resolve(fileName).normalize();
        Path latestFile = baseDir.resolve("obs_latest.json");

        CompletableFuture.runAsync(() -> {
            try {
                Files.createDirectories(baseDir);
                String jsonString = GSON.toJson(root);
                Files.writeString(targetFile, jsonString, StandardCharsets.UTF_8);
                Files.writeString(latestFile, jsonString, StandardCharsets.UTF_8);

                LOGGER.info("Observe: Captured all LODs to {}", targetFile.toAbsolutePath());

                client.execute(() -> {
                    if (client.player != null) {
                        MutableText msg = Text.literal("§a[Observe]§r Captured all 3 LODs (B0, B1, B2) [3x 16x16x16] to ")
                            .append(Text.literal("§6" + targetFile.getFileName() + "§r").styled(style -> style
                                .withUnderline(true)
                                .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, targetFile.toAbsolutePath().toString()))
                            ))
                            .append(Text.literal(" §7[Click to open]§r"));

                        source.sendFeedback(msg);
                    }
                });
            } catch (IOException e) {
                LOGGER.error("Observe: Failed to save file", e);
                client.execute(() -> {
                    source.sendError(Text.literal("§c[Observe] Failed to save file: " + e.getMessage()));
                });
            }
        });

        return Command.SINGLE_SUCCESS;
    }

    /**
     * Precomputes relative 1-meter offsets for indices [-8 .. +7] * stride.
     */
    private static int[] precomputeRelativeOffsets(int stride) {
        int[] offsets = new int[ObservationConfig.GRID_SIZE];
        for (int i = 0; i < ObservationConfig.GRID_SIZE; i++) {
            offsets[i] = (-8 + i) * stride;
        }
        return offsets;
    }

    /**
     * Optimized sampling method:
     * - Uses a single BlockPos.Mutable instance to avoid thousands of allocations.
     * - Reuses a single cell frequency map to eliminate per-cell map creation.
     * - Uses interned Block -> String ID cache.
     */
    private static SamplingResult sampleGrid(
        ClientWorld world,
        BlockPos playerPos,
        int stride,
        int[] relOffsets
    ) {
        final int size = ObservationConfig.GRID_SIZE;
        final int playerX = playerPos.getX();
        final int playerY = playerPos.getY();
        final int playerZ = playerPos.getZ();

        BlockPos.Mutable mutablePos = new BlockPos.Mutable();
        Map<String, Integer> cellCounts = new HashMap<>(32);
        Map<String, Integer> globalCounts = new LinkedHashMap<>();

        JsonArray xArray = new JsonArray();
        int airCells = 0;

        for (int gx = 0; gx < size; gx++) {
            int dx = relOffsets[gx];
            int originX = playerX + dx;

            JsonArray yArray = new JsonArray();
            for (int gy = 0; gy < size; gy++) {
                int dy = relOffsets[gy];
                int originY = playerY + dy;

                JsonArray zArray = new JsonArray();
                for (int gz = 0; gz < size; gz++) {
                    int dz = relOffsets[gz];
                    int originZ = playerZ + dz;

                    String selectedBlock;

                    if (stride == 1) {
                        // LOD 0 fast-path: 1:1 direct sampling
                        mutablePos.set(originX, originY, originZ);
                        BlockState state = world.getBlockState(mutablePos);
                        selectedBlock = getBlockId(state.getBlock());
                    } else {
                        // LOD 1 & 2 downsampling: most frequent block in the stride^3 sub-volume
                        cellCounts.clear();
                        for (int ox = 0; ox < stride; ox++) {
                            for (int oy = 0; oy < stride; oy++) {
                                for (int oz = 0; oz < stride; oz++) {
                                    mutablePos.set(originX + ox, originY + oy, originZ + oz);
                                    BlockState state = world.getBlockState(mutablePos);
                                    String id = getBlockId(state.getBlock());
                                    cellCounts.put(id, cellCounts.getOrDefault(id, 0) + 1);
                                }
                            }
                        }

                        // Determine mode with tie-breaking (prefer non-air)
                        String bestBlock = "minecraft:air";
                        int bestCount = -1;
                        for (Map.Entry<String, Integer> entry : cellCounts.entrySet()) {
                            String blockId = entry.getKey();
                            int count = entry.getValue();

                            if (count > bestCount) {
                                bestCount = count;
                                bestBlock = blockId;
                            } else if (count == bestCount) {
                                boolean currentIsAir = isAirId(bestBlock);
                                boolean candidateIsAir = isAirId(blockId);
                                if (currentIsAir && !candidateIsAir) {
                                    bestBlock = blockId;
                                } else if (!currentIsAir && !candidateIsAir && blockId.compareTo(bestBlock) < 0) {
                                    bestBlock = blockId;
                                }
                            }
                        }
                        selectedBlock = bestBlock;
                    }

                    if (isAirId(selectedBlock)) {
                        airCells++;
                    }
                    globalCounts.put(selectedBlock, globalCounts.getOrDefault(selectedBlock, 0) + 1);

                    // Cell format with relative coordinates in 1-meter units (dx, dy, dz)
                    JsonObject cellObj = new JsonObject();
                    cellObj.addProperty("name", selectedBlock);
                    cellObj.addProperty("dx", dx);
                    cellObj.addProperty("dy", dy);
                    cellObj.addProperty("dz", dz);

                    zArray.add(cellObj);
                }
                yArray.add(zArray);
            }
            xArray.add(yArray);
        }

        int totalCells = size * size * size;
        int nonAirCells = totalCells - airCells;

        return new SamplingResult(xArray, globalCounts, totalCells, airCells, nonAirCells);
    }

    private static JsonObject buildMetadata(
        ClientWorld world,
        BlockPos playerPos,
        int lod,
        int stride,
        int[] relCoords,
        SamplingResult result
    ) {
        JsonObject meta = new JsonObject();
        meta.addProperty("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
        meta.addProperty("dimension", world.getRegistryKey().getValue().toString());
        meta.addProperty("gridSize", ObservationConfig.GRID_SIZE);
        meta.addProperty("lod", lod);
        meta.addProperty("stride", stride);
        meta.addProperty("coordinateFrame", "relative_player_feet");

        int span = ObservationConfig.GRID_SIZE * stride;
        JsonObject coverage = new JsonObject();
        coverage.addProperty("sizeX", span);
        coverage.addProperty("sizeY", span);
        coverage.addProperty("sizeZ", span);
        meta.add("worldCoverage", coverage);

        JsonObject playerObj = new JsonObject();
        playerObj.addProperty("x", playerPos.getX());
        playerObj.addProperty("y", playerPos.getY());
        playerObj.addProperty("z", playerPos.getZ());
        meta.add("playerWorldPos", playerObj);

        JsonObject bounds = new JsonObject();
        JsonObject minRel = new JsonObject();
        minRel.addProperty("dx", relCoords[0]);
        minRel.addProperty("dy", relCoords[0]);
        minRel.addProperty("dz", relCoords[0]);

        JsonObject maxRel = new JsonObject();
        maxRel.addProperty("dx", relCoords[ObservationConfig.GRID_SIZE - 1]);
        maxRel.addProperty("dy", relCoords[ObservationConfig.GRID_SIZE - 1]);
        maxRel.addProperty("dz", relCoords[ObservationConfig.GRID_SIZE - 1]);

        bounds.add("minRelative", minRel);
        bounds.add("maxRelative", maxRel);
        meta.add("bounds", bounds);

        JsonObject stats = new JsonObject();
        stats.addProperty("totalCells", result.totalCells);
        stats.addProperty("totalWorldBlocks", result.totalCells * stride * stride * stride);
        stats.addProperty("airCells", result.airCells);
        stats.addProperty("nonAirCells", result.nonAirCells);

        JsonObject countsObj = new JsonObject();
        result.globalCounts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .forEach(entry -> countsObj.addProperty(entry.getKey(), entry.getValue()));
        stats.add("blockCounts", countsObj);
        meta.add("stats", stats);

        return meta;
    }

    private static String getBlockId(Block block) {
        return BLOCK_ID_CACHE.computeIfAbsent(block, b -> Registries.BLOCK.getId(b).toString());
    }

    private static boolean isAirId(String id) {
        return "minecraft:air".equals(id) || "minecraft:cave_air".equals(id) || "minecraft:void_air".equals(id);
    }

    private record SamplingResult(
        JsonArray gridArray,
        Map<String, Integer> globalCounts,
        int totalCells,
        int airCells,
        int nonAirCells
    ) {}
}
