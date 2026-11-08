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
import org.sqsense.squared_sense_mc.client.observation.entity.EntityGridSampler;
import org.sqsense.squared_sense_mc.client.observation.entity.EntitySamplingResult;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.geometry.YawHysteresisTracker;
import org.sqsense.squared_sense_mc.client.observation.ipc.ObservationStreamManager;
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

        var intervalArg = ClientCommandManager.argument("interval", IntegerArgumentType.integer(1, 100))
            .executes(ctx -> executeStreamStart(ctx, IntegerArgumentType.getInteger(ctx, "interval")));

        var streamStart = ClientCommandManager.literal("start")
            .executes(ctx -> executeStreamStart(ctx, ObservationConfig.DEFAULT_STREAM_INTERVAL_TICKS))
            .then(intervalArg);

        var streamStop = ClientCommandManager.literal("stop")
            .executes(ObserveCommand::executeStreamStop);

        var streamStatus = ClientCommandManager.literal("status")
            .executes(ObserveCommand::executeStreamStatus);

        var streamLiteral = ClientCommandManager.literal("stream")
            .executes(ObserveCommand::executeStreamStatus)
            .then(streamStart)
            .then(streamStop)
            .then(streamStatus);

        var visOn = ClientCommandManager.literal("on")
            .executes(ctx -> executeSetVisibility(ctx, true));
        var visOff = ClientCommandManager.literal("off")
            .executes(ctx -> executeSetVisibility(ctx, false));
        var visStatus = ClientCommandManager.literal("status")
            .executes(ObserveCommand::executeVisibilityStatus);

        var visibilityLiteral = ClientCommandManager.literal("visibility")
            .executes(ObserveCommand::executeToggleVisibility)
            .then(visOn)
            .then(visOff)
            .then(visStatus);

        var observeNode = dispatcher.register(
            ClientCommandManager.literal("observe")
                .executes(ctx -> executeSingle(ctx, 0, null))
                .then(visibilityLiteral)
                .then(streamLiteral)
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

        // Sample 16x16x16 block grid
        SamplingResult blockResult = BlockSampler.sample(world, playerPos, transform);

        // Sample 16x16x16 entity grid (Task 10)
        EntitySamplingResult entityResult = EntityGridSampler.sample(world, player, playerPos, transform);

        // Build JSON representation
        JsonObject root = ObservationJsonSerializer.buildSingleLodJson(
            world,
            playerPos,
            lod,
            transform,
            blockResult,
            entityResult
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
                        String entitySummary = entityResult.totalEntities() > 0
                            ? ", §d" + entityResult.totalEntities() + " entities§r [§c" + entityResult.hostileEntities() + " hostile§r]"
                            : ", §70 entities§r";
                        MutableText msg = Text.literal("§a[Observe]§r Captured LOD §b" + lod + "§r ("
                            + cardinal.name() + " @ " + cardinal.getYawDegrees() + "°, pitch footIndex " + transform.getFootIndex()
                            + ", stride " + stride + ", spans §e" + span + "x" + span + "x" + span + "m§r" + entitySummary + ") to ")
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

        EntitySamplingResult ent0 = EntityGridSampler.sample(world, player, playerPos, transform0);
        EntitySamplingResult ent1 = EntityGridSampler.sample(world, player, playerPos, transform1);
        EntitySamplingResult ent2 = EntityGridSampler.sample(world, player, playerPos, transform2);

        JsonObject root = ObservationJsonSerializer.buildAllLodsJson(
            world,
            playerPos,
            transform0,
            res0,
            res1,
            res2,
            ent0,
            ent1,
            ent2
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
                        int totalFound = ent2.totalEntities();
                        String entitySummary = totalFound > 0
                            ? ", §d" + totalFound + " entities in B2§r [§c" + ent2.hostileEntities() + " hostile§r]"
                            : ", §70 entities§r";
                        MutableText msg = Text.literal("§a[Observe]§r Captured all 3 LODs ("
                            + cardinal.name() + " @ " + cardinal.getYawDegrees() + "°, pitch footIndex " + transform0.getFootIndex()
                            + entitySummary + ") to ")
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

    private static int executeStreamStart(CommandContext<FabricClientCommandSource> context, int interval) {
        var source = context.getSource();
        ObservationStreamManager manager = ObservationStreamManager.getInstance();

        boolean success = manager.start(interval);
        if (success) {
            float hz = 20.0f / interval;
            Path shmPath = manager.getChannel().getFilePath();
            MutableText msg = Text.literal("§a[Observe]§r Started real-time MMF stream every §b"
                + interval + " ticks§r (§e" + String.format("%.1f", hz) + " Hz§r) to ")
                .append(Text.literal("§6" + shmPath.getFileName() + "§r").styled(style -> style
                    .withUnderline(true)
                    .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, shmPath.toAbsolutePath().toString()))
                ))
                .append(Text.literal(" §7[Click to open]§r"));
            source.sendFeedback(msg);
            return Command.SINGLE_SUCCESS;
        } else {
            source.sendError(Text.literal("§c[Observe] Failed to open MMF shared memory channel."));
            return 0;
        }
    }

    private static int executeStreamStop(CommandContext<FabricClientCommandSource> context) {
        var source = context.getSource();
        ObservationStreamManager manager = ObservationStreamManager.getInstance();

        long frames = manager.getFramesStreamed();
        manager.stop();

        source.sendFeedback(Text.literal("§a[Observe]§r Stopped real-time MMF stream. (Total frames streamed: §b" + frames + "§r)"));
        return Command.SINGLE_SUCCESS;
    }

    private static int executeStreamStatus(CommandContext<FabricClientCommandSource> context) {
        var source = context.getSource();
        ObservationStreamManager manager = ObservationStreamManager.getInstance();

        if (manager.isActive()) {
            int interval = manager.getIntervalTicks();
            float hz = 20.0f / interval;
            Path shmPath = manager.getChannel().getFilePath();
            MutableText msg = Text.literal("§a[Observe]§r Stream is §2ACTIVE§r: interval §b"
                + interval + " ticks§r (§e" + String.format("%.1f", hz) + " Hz§r), §b"
                + manager.getFramesStreamed() + " frames§r streamed to ")
                .append(Text.literal("§6" + shmPath.getFileName() + "§r").styled(style -> style
                    .withUnderline(true)
                    .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_FILE, shmPath.toAbsolutePath().toString()))
                ));
            source.sendFeedback(msg);
        } else {
            source.sendFeedback(Text.literal("§a[Observe]§r Stream is §7INACTIVE§r. Use §e/observe stream start [interval]§r to begin streaming."));
        }
        return Command.SINGLE_SUCCESS;
    }

    private static int executeToggleVisibility(CommandContext<FabricClientCommandSource> context) {
        boolean enabled = ObservationConfig.toggleVisibilityFilter();
        context.getSource().sendFeedback(Text.literal(
            "§b[SquaredSense] §fVisibility / Occlusion Filter (Phase 5): " +
            (enabled ? "§aENABLED (culling solid underworld)" : "§cDISABLED (omniscient full underworld)")
        ));
        return Command.SINGLE_SUCCESS;
    }

    private static int executeSetVisibility(CommandContext<FabricClientCommandSource> context, boolean enabled) {
        ObservationConfig.visibilityFilterEnabled = enabled;
        context.getSource().sendFeedback(Text.literal(
            "§b[SquaredSense] §fVisibility / Occlusion Filter (Phase 5): " +
            (enabled ? "§aENABLED (culling solid underworld)" : "§cDISABLED (omniscient full underworld)")
        ));
        return Command.SINGLE_SUCCESS;
    }

    private static int executeVisibilityStatus(CommandContext<FabricClientCommandSource> context) {
        boolean enabled = ObservationConfig.visibilityFilterEnabled;
        context.getSource().sendFeedback(Text.literal(
            "§b[SquaredSense] §fVisibility / Occlusion Filter (Phase 5): " +
            (enabled ? "§aENABLED (culling solid underworld)" : "§cDISABLED (omniscient full underworld)")
        ));
        return Command.SINGLE_SUCCESS;
    }

    private static String resolveFilename(String customFilename, String defaultFilename) {
        if (customFilename != null && !customFilename.isBlank()) {
            String sanitized = customFilename.trim().replaceAll("[\\\\/:*?\"<>|]", "_");
            return sanitized.endsWith(".json") ? sanitized : sanitized + ".json";
        }
        return defaultFilename;
    }
}
