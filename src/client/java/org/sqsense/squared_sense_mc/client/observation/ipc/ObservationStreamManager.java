package org.sqsense.squared_sense_mc.client.observation.ipc;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.geometry.YawHysteresisTracker;

/**
 * Manages the continuous real-time MMF observation stream:
 * - Runs on ClientTickEvents.END_CLIENT_TICK.
 * - Samples B0..B2 and E0..E2 on a configurable cadence (default: every 2 ticks / 10 Hz).
 * - Writes directly into off-heap memory with sub-millisecond execution.
 */
public class ObservationStreamManager {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");
    private static ObservationStreamManager instance;

    private final SharedMemoryChannel channel = SharedMemoryChannel.getInstance();
    private final BinaryObservationWriter writer = new BinaryObservationWriter();
    private final YawHysteresisTracker yawTracker = new YawHysteresisTracker();

    private boolean active = false;
    private int intervalTicks = ObservationConfig.DEFAULT_STREAM_INTERVAL_TICKS;
    private long tickCounter = 0;
    private long framesStreamed = 0;
    private long lastStreamTimeNanos = 0;

    private ObservationStreamManager() {}

    public static synchronized ObservationStreamManager getInstance() {
        if (instance == null) {
            instance = new ObservationStreamManager();
        }
        return instance;
    }

    /**
     * Starts continuous MMF streaming at the specified tick cadence.
     */
    public synchronized boolean start(int intervalTicks) {
        if (intervalTicks < 1) {
            intervalTicks = 1;
        }

        if (!channel.open()) {
            LOGGER.error("ObservationStreamManager: Cannot start streaming because MMF channel failed to open.");
            return false;
        }

        this.intervalTicks = intervalTicks;
        this.active = true;
        this.framesStreamed = 0;
        this.tickCounter = 0;

        float hz = 20.0f / intervalTicks;
        LOGGER.info("ObservationStreamManager: Started streaming every {} ticks ({} Hz) to {}",
            intervalTicks, hz, channel.getFilePath().toAbsolutePath());
        return true;
    }

    /**
     * Stops continuous streaming.
     */
    public synchronized void stop() {
        this.active = false;
        LOGGER.info("ObservationStreamManager: Stopped streaming. (Total frames streamed: {})", framesStreamed);
    }

    /**
     * Client tick callback invoked from ClientTickEvents.END_CLIENT_TICK.
     */
    public void onClientTick(MinecraftClient client) {
        if (!active || client.world == null || client.player == null) {
            return;
        }

        tickCounter++;
        if (tickCounter % intervalTicks != 0) {
            return;
        }

        ClientWorld world = client.world;
        ClientPlayerEntity player = client.player;
        BlockPos playerPos = player.getBlockPos();
        SnappedCardinal cardinal = yawTracker.update(player.getYaw());
        long gameTick = world.getTime();

        writer.writeObservation(
            world,
            player,
            playerPos,
            cardinal,
            player.getYaw(),
            player.getPitch(),
            gameTick,
            channel
        );

        framesStreamed++;
        lastStreamTimeNanos = System.nanoTime();
    }

    public boolean isActive() {
        return active;
    }

    public int getIntervalTicks() {
        return intervalTicks;
    }

    public long getFramesStreamed() {
        return framesStreamed;
    }

    public long getLastStreamTimeNanos() {
        return lastStreamTimeNanos;
    }

    public SharedMemoryChannel getChannel() {
        return channel;
    }
}

