package org.sqsense.squared_sense_mc.client.observation.ipc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;

import java.io.File;
import java.io.RandomAccessFile;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-platform Memory-Mapped File (MMF) IPC channel for ultra-low latency observation streaming.
 *
 * Supported platforms:
 * - Linux: /dev/shm/squared_sense_mc.bin (tmpfs RAM backing, zero disk touches)
 * - Windows: %TEMP%/squared_sense_mc.bin (System Page Cache in RAM, sub-millisecond bus speed)
 *
 * Employs Ping-Pong Double Buffering with memory fences to ensure zero torn reads and
 * zero lock contention between Minecraft's render thread and external consumers.
 */
public class SharedMemoryChannel {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");
    private static SharedMemoryChannel instance;

    private final Path filePath;
    private RandomAccessFile raf;
    private FileChannel fileChannel;
    private MappedByteBuffer buffer;

    private int activeSlot = 0;
    private long sequenceCounter = 0;
    private boolean open = false;

    private SharedMemoryChannel(Path filePath) {
        this.filePath = filePath;
    }

    public static synchronized SharedMemoryChannel getInstance() {
        if (instance == null) {
            instance = new SharedMemoryChannel(resolveDefaultPath());
        }
        return instance;
    }

    public static Path resolveDefaultPath() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("linux")) {
            Path shmDir = Path.of("/dev/shm");
            if (Files.isDirectory(shmDir) && Files.isWritable(shmDir)) {
                return shmDir.resolve("squared_sense_mc.bin");
            }
        }
        String tempDir = System.getProperty("java.io.tmpdir");
        return Path.of(tempDir).resolve("squared_sense_mc.bin");
    }

    /**
     * Initializes and maps the shared memory file.
     * Guaranteed to be idempotent and safe across restarts.
     */
    public synchronized boolean open() {
        if (open) {
            return true;
        }

        try {
            if (filePath.getParent() != null) {
                Files.createDirectories(filePath.getParent());
            }

            File file = filePath.toFile();
            this.raf = new RandomAccessFile(file, "rw");
            this.raf.setLength(IpcBufferLayout.TOTAL_FILE_SIZE);

            this.fileChannel = raf.getChannel();
            this.buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, IpcBufferLayout.TOTAL_FILE_SIZE);
            this.buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Initialize Page 0 header
            initHeader();

            this.open = true;
            LOGGER.info("SharedMemoryChannel: Successfully mapped {} bytes at {}",
                IpcBufferLayout.TOTAL_FILE_SIZE, filePath.toAbsolutePath());
            return true;
        } catch (Exception e) {
            LOGGER.error("SharedMemoryChannel: Failed to open MMF at {}", filePath.toAbsolutePath(), e);
            close();
            return false;
        }
    }

    private void initHeader() {
        buffer.putInt(IpcBufferLayout.OFF_MAGIC, IpcBufferLayout.MAGIC);
        buffer.putInt(IpcBufferLayout.OFF_VERSION, IpcBufferLayout.PROTOCOL_VERSION);
        buffer.putInt(IpcBufferLayout.OFF_HEADER_SIZE, IpcBufferLayout.HEADER_SIZE);
        buffer.putInt(IpcBufferLayout.OFF_SLOT_SIZE, IpcBufferLayout.SLOT_SIZE);
        buffer.putLong(IpcBufferLayout.OFF_SEQUENCE_ID, sequenceCounter);
        buffer.putInt(IpcBufferLayout.OFF_ACTIVE_SLOT, activeSlot);
        buffer.putInt(IpcBufferLayout.OFF_FLAGS, 1); // 1 = streaming channel active
        buffer.putInt(IpcBufferLayout.OFF_VOCAB_SIZE, BlockVocabulary.getVocabSize());
        buffer.put(IpcBufferLayout.OFF_STRIDE_0, (byte) ObservationConfig.getStride(0));
        buffer.put(IpcBufferLayout.OFF_STRIDE_1, (byte) ObservationConfig.getStride(1));
        buffer.put(IpcBufferLayout.OFF_STRIDE_2, (byte) ObservationConfig.getStride(2));
    }

    /**
     * Atomically publishes the freshly written slot to consumers.
     */
    public synchronized void commitSlot(
        int slotIndex,
        long gameTick,
        double px,
        double py,
        double pz,
        float yaw,
        float pitch,
        SnappedCardinal cardinal,
        int footIndex
    ) {
        if (!open || buffer == null) {
            return;
        }

        sequenceCounter++;

        // 1. Update pose and timestamp fields in header
        buffer.putLong(IpcBufferLayout.OFF_GAME_TICK, gameTick);
        buffer.putLong(IpcBufferLayout.OFF_TIMESTAMP_NANOS, System.nanoTime());
        buffer.putDouble(IpcBufferLayout.OFF_PLAYER_X, px);
        buffer.putDouble(IpcBufferLayout.OFF_PLAYER_Y, py);
        buffer.putDouble(IpcBufferLayout.OFF_PLAYER_Z, pz);
        buffer.putFloat(IpcBufferLayout.OFF_CAMERA_YAW, yaw);
        buffer.putFloat(IpcBufferLayout.OFF_CAMERA_PITCH, pitch);
        buffer.putFloat(IpcBufferLayout.OFF_SNAPPED_YAW, cardinal.getYawDegrees());
        buffer.put(IpcBufferLayout.OFF_CARDINAL, (byte) cardinal.ordinal());
        buffer.put(IpcBufferLayout.OFF_FOOT_INDEX, (byte) footIndex);
        buffer.put(IpcBufferLayout.OFF_FORWARD_INDEX, (byte) ObservationConfig.PLAYER_FORWARD_INDEX);
        buffer.put(IpcBufferLayout.OFF_LATERAL_INDEX, (byte) ObservationConfig.PLAYER_LATERAL_INDEX);
        buffer.putInt(IpcBufferLayout.OFF_VOCAB_SIZE, BlockVocabulary.getVocabSize());

        // 2. Memory store barrier: ensures all tensor payload writes and header updates
        // are globally committed before the activeSlot index is flipped.
        VarHandle.releaseFence();

        // 3. Atomically flip active slot and publish new sequence ID
        this.activeSlot = slotIndex & 1;
        buffer.putInt(IpcBufferLayout.OFF_ACTIVE_SLOT, this.activeSlot);
        buffer.putLong(IpcBufferLayout.OFF_SEQUENCE_ID, sequenceCounter);
    }

    public synchronized void close() {
        open = false;
        buffer = null;

        if (fileChannel != null) {
            try {
                fileChannel.close();
            } catch (Exception ignored) {}
            fileChannel = null;
        }

        if (raf != null) {
            try {
                raf.close();
            } catch (Exception ignored) {}
            raf = null;
        }

        LOGGER.info("SharedMemoryChannel: Closed IPC channel at {}", filePath.toAbsolutePath());
    }

    public boolean isOpen() {
        return open;
    }

    public int getActiveSlot() {
        return activeSlot;
    }

    public int getInactiveSlot() {
        return 1 - activeSlot;
    }

    public MappedByteBuffer getBuffer() {
        return buffer;
    }

    public Path getFilePath() {
        return filePath;
    }

    public long getSequenceCounter() {
        return sequenceCounter;
    }
}

