package org.sqsense.squared_sense_mc.client.command;

import com.google.gson.JsonObject;
import com.mojang.brigadier.Command;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.text.ClickEvent;
import net.minecraft.text.MutableText;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.geometry.YawHysteresisTracker;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;
import org.sqsense.squared_sense_mc.client.observation.sampler.BlockSampler;
import org.sqsense.squared_sense_mc.client.observation.sampler.SamplingResult;
import org.sqsense.squared_sense_mc.client.observation.serialization.ObservationJsonSerializer;

import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Command controller for /observe and /obs.
 * Integrates cardinal yaw snapping with hysteresis, asymmetric forward offset,
 * vertical pitch translation, and multi-LOD 16x16x16 sampling.
 */
public class ObserveCommand {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");
    private static final YawHysteresisTracker YAW_TRACKER = new YawHysteresisTracker();

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

        var vocabLiteral = ClientCommandManager.literal("vocab")
            .executes(ObserveCommand::executeExportVocab);

        var observeNode = dispatcher.register(
            ClientCommandManager.literal("observe")
                .executes(ctx -> executeSingle(ctx, 0, null))
                .then(vocabLiteral)
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

        // Update yaw with hysteresis deadband (+/-10 deg around 45 deg diagonals)
        SnappedCardinal cardinal = YAW_TRACKER.update(player.getYaw());

        // Spatial transform incorporating yaw rotation, asymmetric forward offset, and pitch translation
        ObservationTransform transform = new ObservationTransform(
            player.getYaw(),
            player.getPitch(),
            cardinal,
            stride
        );

        // Sample 16x16x16 grid
        SamplingResult result = BlockSampler.sample(world, playerPos, transform);

        // Build JSON representation
        JsonObject root = ObservationJsonSerializer.buildSingleLodJson(
            world,
            playerPos,
            lod,
            transform,
            result
        );

        // Determine destination paths
        Path baseDir = client.runDirectory.toPath().resolve("observations");
        String fileName = resolveFilename(customFilename, String.format(
            "obs_lod%d_%s.json",
            lod,
            DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss").format(LocalDateTime.now())
        ));

        Path targetFile = baseDir.resolve(fileName).normalize();
        Path latestFile = baseDir.resolve("obs_latest.json");
        Path latestLodFile = baseDir.resolve(String.format("obs_lod%d_latest.json", lod));

        // Asynchronous file write
        ObservationJsonSerializer.writeAsync(
            root,
            targetFile,
            latestFile,
            latestLodFile,
            savedPath -> {
                LOGGER.info("Observe: Saved LOD {} to {}", lod, savedPath.toAbsolutePath());
                client.execute(() -> {
                    if (client.player != null) {
                        int span = ObservationConfig.GRID_SIZE * stride;
                        MutableText msg = Text.literal("§a[Observe]§r Captured LOD §b" + lod + "§r ("
                            + cardinal.name() + " @ " + cardinal.getYawDegrees() + "°, pitch footIndex " + transform.getFootIndex()
                            + ", stride " + stride + ", spans §e" + span + "x" + span + "x" + span + "m§r) to ")
                            .append(Text.literal("§6" + targetFile.getFileName() + "§r").styled(style -> style
                                .withUnderline(true)
                                .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, targetFile.toAbsolutePath().toString()))
                            ))
                            .append(Text.literal(" §7[Click to open]§r"));

                        source.sendFeedback(msg);
                    }
                });
            },
            error -> {
                LOGGER.error("Observe: Failed to save file", error);
                client.execute(() -> {
                    source.sendError(Text.literal("§c[Observe] Failed to save file: " + error.getMessage()));
                });
            }
        );

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
        SnappedCardinal cardinal = YAW_TRACKER.update(player.getYaw());

        ObservationTransform transform0 = new ObservationTransform(player.getYaw(), player.getPitch(), cardinal, ObservationConfig.getStride(0));
        ObservationTransform transform1 = new ObservationTransform(player.getYaw(), player.getPitch(), cardinal, ObservationConfig.getStride(1));
        ObservationTransform transform2 = new ObservationTransform(player.getYaw(), player.getPitch(), cardinal, ObservationConfig.getStride(2));

        SamplingResult res0 = BlockSampler.sample(world, playerPos, transform0);
        SamplingResult res1 = BlockSampler.sample(world, playerPos, transform1);
        SamplingResult res2 = BlockSampler.sample(world, playerPos, transform2);

        JsonObject root = ObservationJsonSerializer.buildAllLodsJson(
            world,
            playerPos,
            transform0,
            res0,
            res1,
            res2
        );

        Path baseDir = client.runDirectory.toPath().resolve("observations");
        String fileName = resolveFilename(customFilename, String.format(
            "obs_all_%s.json",
            DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss").format(LocalDateTime.now())
        ));

        Path targetFile = baseDir.resolve(fileName).normalize();
        Path latestFile = baseDir.resolve("obs_latest.json");

        ObservationJsonSerializer.writeAsync(
            root,
            targetFile,
            latestFile,
            null,
            savedPath -> {
                LOGGER.info("Observe: Saved all LODs to {}", savedPath.toAbsolutePath());
                client.execute(() -> {
                    if (client.player != null) {
                        MutableText msg = Text.literal("§a[Observe]§r Captured all 3 LODs ("
                            + cardinal.name() + " @ " + cardinal.getYawDegrees() + "°, pitch footIndex " + transform0.getFootIndex()
                            + ") to ")
                            .append(Text.literal("§6" + targetFile.getFileName() + "§r").styled(style -> style
                                .withUnderline(true)
                                .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, targetFile.toAbsolutePath().toString()))
                            ))
                            .append(Text.literal(" §7[Click to open]§r"));

                        source.sendFeedback(msg);
                    }
                });
            },
            error -> {
                LOGGER.error("Observe: Failed to save file", error);
                client.execute(() -> {
                    source.sendError(Text.literal("§c[Observe] Failed to save file: " + error.getMessage()));
                });
            }
        );

        return Command.SINGLE_SUCCESS;
    }

    private static int executeExportVocab(CommandContext<FabricClientCommandSource> context) {
        var source = context.getSource();
        MinecraftClient client = source.getClient();
        if (client == null) {
            source.sendError(Text.literal("§c[Observe] Client not available."));
            return 0;
        }

        Path baseDir = client.runDirectory.toPath().resolve("observations");
        Path vocabFile = baseDir.resolve("block_vocab.json");

        try {
            BlockVocabulary.exportVocabulary(vocabFile);
            int size = BlockVocabulary.getVocabSize();
            MutableText msg = Text.literal("§a[Observe]§r Exported canonical vocabulary (§b" + size + "§r tokens) to ")
                .append(Text.literal("§6block_vocab.json§r").styled(style -> style
                    .withUnderline(true)
                    .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, vocabFile.toAbsolutePath().toString()))
                ))
                .append(Text.literal(" §7[Click to open]§r"));
            source.sendFeedback(msg);
            LOGGER.info("Observe: Exported block vocabulary ({} tokens) to {}", size, vocabFile.toAbsolutePath());
            return Command.SINGLE_SUCCESS;
        } catch (Exception e) {
            LOGGER.error("Observe: Failed to export vocabulary", e);
            source.sendError(Text.literal("§c[Observe] Failed to export vocabulary: " + e.getMessage()));
            return 0;
        }
    }

    private static String resolveFilename(String customFilename, String defaultFilename) {
        if (customFilename != null && !customFilename.isBlank()) {
            String sanitized = customFilename.trim().replaceAll("[\\\\/:*?\"<>|]", "_");
            return sanitized.endsWith(".json") ? sanitized : sanitized + ".json";
        }
        return defaultFilename;
    }
}
