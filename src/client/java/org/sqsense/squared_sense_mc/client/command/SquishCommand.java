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

public class SquishCommand {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final int MAX_WORLD_SPAN = 256;

    public static void register(CommandDispatcher<FabricClientCommandSource> dispatcher) {
        var filenameArg = ClientCommandManager.argument("filename", StringArgumentType.string())
            .executes(ctx -> execute(
                ctx,
                IntegerArgumentType.getInteger(ctx, "size"),
                IntegerArgumentType.getInteger(ctx, "lod"),
                StringArgumentType.getString(ctx, "filename")
            ));

        var lodArg = ClientCommandManager.argument("lod", IntegerArgumentType.integer(1, 16))
            .executes(ctx -> execute(
                ctx,
                IntegerArgumentType.getInteger(ctx, "size"),
                IntegerArgumentType.getInteger(ctx, "lod"),
                null
            ))
            .then(filenameArg);

        var sizeArg = ClientCommandManager.argument("size", IntegerArgumentType.integer(1, 64))
            .executes(ctx -> execute(ctx, IntegerArgumentType.getInteger(ctx, "size"), 1, null))
            .then(lodArg);

        dispatcher.register(
            ClientCommandManager.literal("squish")
                .executes(ctx -> execute(ctx, 3, 1, null))
                .then(sizeArg)
        );
    }

    private static int execute(
        CommandContext<FabricClientCommandSource> context,
        int size,
        int lod,
        String customFilename
    ) {
        var source = context.getSource();
        MinecraftClient client = source.getClient();
        ClientPlayerEntity player = client.player;
        ClientWorld world = client.world;

        if (client == null || player == null || world == null) {
            source.sendError(Text.literal("§c[Squish] World or player not available."));
            return 0;
        }

        int totalSpan = size * lod;
        if (totalSpan > MAX_WORLD_SPAN) {
            source.sendError(Text.literal(
                "§c[Squish] Total coverage (size * lod = " + totalSpan + ") exceeds maximum limit of " + MAX_WORLD_SPAN + " blocks."
            ));
            return 0;
        }

        BlockPos playerPos = player.getBlockPos();
        int offset = (totalSpan - 1) / 2;
        int minX = playerPos.getX() - offset;
        int minY = playerPos.getY() - offset;
        int minZ = playerPos.getZ() - offset;
        int maxX = minX + totalSpan - 1;
        int maxY = minY + totalSpan - 1;
        int maxZ = minZ + totalSpan - 1;

        JsonArray xArray = new JsonArray();
        Map<String, Integer> blockCounts = new LinkedHashMap<>();
        int totalCells = size * size * size;
        int airCells = 0;

        for (int gx = 0; gx < size; gx++) {
            int subMinX = minX + gx * lod;
            int subMaxX = subMinX + lod - 1;

            JsonArray yArray = new JsonArray();
            for (int gy = 0; gy < size; gy++) {
                int subMinY = minY + gy * lod;
                int subMaxY = subMinY + lod - 1;

                JsonArray zArray = new JsonArray();
                for (int gz = 0; gz < size; gz++) {
                    int subMinZ = minZ + gz * lod;
                    int subMaxZ = subMinZ + lod - 1;

                    String selectedBlock;
                    if (lod == 1) {
                        BlockPos pos = new BlockPos(subMinX, subMinY, subMinZ);
                        BlockState state = world.getBlockState(pos);
                        selectedBlock = Registries.BLOCK.getId(state.getBlock()).toString();
                    } else {
                        Map<String, Integer> cellCounts = new HashMap<>();
                        for (int x = subMinX; x <= subMaxX; x++) {
                            for (int y = subMinY; y <= subMaxY; y++) {
                                for (int z = subMinZ; z <= subMaxZ; z++) {
                                    BlockPos pos = new BlockPos(x, y, z);
                                    BlockState state = world.getBlockState(pos);
                                    String id = Registries.BLOCK.getId(state.getBlock()).toString();
                                    cellCounts.put(id, cellCounts.getOrDefault(id, 0) + 1);
                                }
                            }
                        }

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
                    blockCounts.put(selectedBlock, blockCounts.getOrDefault(selectedBlock, 0) + 1);

                    JsonObject blockObj = new JsonObject();
                    blockObj.addProperty("name", selectedBlock);
                    blockObj.addProperty("x", subMinX);
                    blockObj.addProperty("y", subMinY);
                    blockObj.addProperty("z", subMinZ);

                    zArray.add(blockObj);
                }
                yArray.add(zArray);
            }
            xArray.add(yArray);
        }

        int nonAirCells = totalCells - airCells;

        JsonObject root = new JsonObject();

        JsonObject meta = new JsonObject();
        meta.addProperty("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
        meta.addProperty("dimension", world.getRegistryKey().getValue().toString());
        meta.addProperty("size", size);
        meta.addProperty("lod", lod);

        JsonObject coverage = new JsonObject();
        coverage.addProperty("sizeX", totalSpan);
        coverage.addProperty("sizeY", totalSpan);
        coverage.addProperty("sizeZ", totalSpan);
        meta.add("worldCoverage", coverage);

        JsonObject playerObj = new JsonObject();
        playerObj.addProperty("x", playerPos.getX());
        playerObj.addProperty("y", playerPos.getY());
        playerObj.addProperty("z", playerPos.getZ());
        meta.add("playerPos", playerObj);

        JsonObject minObj = new JsonObject();
        minObj.addProperty("x", minX);
        minObj.addProperty("y", minY);
        minObj.addProperty("z", minZ);

        JsonObject maxObj = new JsonObject();
        maxObj.addProperty("x", maxX);
        maxObj.addProperty("y", maxY);
        maxObj.addProperty("z", maxZ);

        JsonObject bounds = new JsonObject();
        bounds.add("min", minObj);
        bounds.add("max", maxObj);
        meta.add("bounds", bounds);

        JsonObject dimensions = new JsonObject();
        dimensions.addProperty("x", size);
        dimensions.addProperty("y", size);
        dimensions.addProperty("z", size);
        meta.add("dimensions", dimensions);

        JsonObject stats = new JsonObject();
        stats.addProperty("totalCells", totalCells);
        stats.addProperty("totalWorldBlocks", totalSpan * totalSpan * totalSpan);
        stats.addProperty("airCells", airCells);
        stats.addProperty("nonAirCells", nonAirCells);

        JsonObject countsObj = new JsonObject();
        blockCounts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .forEach(entry -> countsObj.addProperty(entry.getKey(), entry.getValue()));
        stats.add("blockCounts", countsObj);
        meta.add("stats", stats);

        root.add("metadata", meta);
        root.add("grid", xArray);

        Path baseDir = client.runDirectory.toPath().resolve("squish_output");
        String fileName;
        if (customFilename != null && !customFilename.isBlank()) {
            String sanitized = customFilename.trim().replaceAll("[\\\\/:*?\"<>|]", "_");
            fileName = sanitized.endsWith(".json") ? sanitized : sanitized + ".json";
        } else {
            String timestamp = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss").format(LocalDateTime.now());
            if (lod == 1) {
                fileName = String.format("squish_%dx%dx%d_%s.json", size, size, size, timestamp);
            } else {
                fileName = String.format("squish_%dx%dx%d_lod%d_%s.json", size, size, size, lod, timestamp);
            }
        }

        Path targetFile = baseDir.resolve(fileName).normalize();
        Path latestFile = baseDir.resolve("squish_latest.json");

        CompletableFuture.runAsync(() -> {
            try {
                Files.createDirectories(baseDir);
                String jsonString = GSON.toJson(root);
                Files.writeString(targetFile, jsonString, StandardCharsets.UTF_8);
                Files.writeString(latestFile, jsonString, StandardCharsets.UTF_8);

                LOGGER.info("Squish: Saved {}x{}x{} grid (LOD {}) to {}", size, size, size, lod, targetFile.toAbsolutePath());

                client.execute(() -> {
                    if (client.player != null) {
                        String lodInfo = lod > 1
                            ? " (LOD §b" + lod + "§r, covers §e" + totalSpan + "x" + totalSpan + "x" + totalSpan + "§r world blocks)"
                            : "";

                        MutableText msg = Text.literal("§a[Squish]§r Saved §e" + size + "x" + size + "x" + size + "§r grid"
                            + lodInfo + " (" + totalCells + " cells, §b" + nonAirCells + "§r solid) to ")
                            .append(Text.literal("§6" + targetFile.getFileName() + "§r").styled(style -> style
                                .withUnderline(true)
                                .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, targetFile.toAbsolutePath().toString()))
                            ))
                            .append(Text.literal(" §7[Click to open]§r"));

                        source.sendFeedback(msg);
                    }
                });
            } catch (IOException e) {
                LOGGER.error("Squish: Failed to save file", e);
                client.execute(() -> {
                    source.sendError(Text.literal("§c[Squish] Failed to save file: " + e.getMessage()));
                });
            }
        });

        return Command.SINGLE_SUCCESS;
    }

    private static boolean isAirId(String id) {
        return "minecraft:air".equals(id) || "minecraft:cave_air".equals(id) || "minecraft:void_air".equals(id);
    }
}
