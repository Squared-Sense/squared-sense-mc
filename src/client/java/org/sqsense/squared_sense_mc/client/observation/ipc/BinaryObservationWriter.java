package org.sqsense.squared_sense_mc.client.observation.ipc;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.util.math.BlockPos;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.channels.PhysicalPropertyExtractor;
import org.sqsense.squared_sense_mc.client.observation.channels.VoxelPhysicalProperties;
import org.sqsense.squared_sense_mc.client.observation.entity.EntityClassifier;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriority;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriorityRegistry;
import org.sqsense.squared_sense_mc.client.observation.visibility.SurfaceShellFilter;

import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Ultra-high-speed binary observation writer for real-time MMF streaming.
 * Writes raw structured binary tensors directly into off-heap memory with ZERO garbage collection overhead:
 * - Samples B0, B1, B2 block grids (token IDs + 7 continuous physical channels).
 * - Samples E0, E1, E2 entity grids (entity counts, max-pooled hostility, total health).
 * - Commits writes atomically using Ping-Pong double-buffering.
 */
public class BinaryObservationWriter {
    private final BlockPos.Mutable mutablePos = new BlockPos.Mutable();
    private final BlockPos.Mutable neighborPos = new BlockPos.Mutable();
    private final Map<Block, Integer> cellCounts = new HashMap<>(32);
    private final PhysicalPropertyExtractor.Accumulator accumulator = new PhysicalPropertyExtractor.Accumulator();

    private final int[] entityCounts = new int[IpcBufferLayout.CELLS_PER_LOD];
    private final byte[] entityTypeIds = new byte[IpcBufferLayout.CELLS_PER_LOD];
    private final float[] entityHostile = new float[IpcBufferLayout.CELLS_PER_LOD];
    private final float[] entityHealth = new float[IpcBufferLayout.CELLS_PER_LOD];

    public void writeObservation(
        ClientWorld world,
        ClientPlayerEntity player,
        BlockPos playerPos,
        SnappedCardinal cardinal,
        float rawYaw,
        float rawPitch,
        long gameTick,
        SharedMemoryChannel channel
    ) {
        if (!channel.isOpen()) {
            return;
        }

        MappedByteBuffer buffer = channel.getBuffer();
        if (buffer == null) {
            return;
        }

        int slotIndex = channel.getInactiveSlot();
        int slotOffset = IpcBufferLayout.getSlotOffset(slotIndex);

        ObservationTransform[] transforms = new ObservationTransform[] {
            new ObservationTransform(rawYaw, rawPitch, cardinal, ObservationConfig.getStride(0)),
            new ObservationTransform(rawYaw, rawPitch, cardinal, ObservationConfig.getStride(1)),
            new ObservationTransform(rawYaw, rawPitch, cardinal, ObservationConfig.getStride(2))
        };

        // 1. Sample and write Block Tensors (B0, B1, B2)
        for (int lod = 0; lod < 3; lod++) {
            writeBlockLod(world, playerPos, transforms[lod], buffer, slotOffset, lod);
        }

        // 2. Sample and write Entity Tensors (E0, E1, E2)
        for (int lod = 0; lod < 3; lod++) {
            writeEntityLod(world, player, playerPos, transforms[lod], buffer, slotOffset, lod);
        }

        // 3. Atomically publish the committed slot
        channel.commitSlot(
            slotIndex,
            gameTick,
            player.getX(),
            player.getY(),
            player.getZ(),
            rawYaw,
            rawPitch,
            cardinal,
            transforms[0].getFootIndex()
        );
    }

    private void writeBlockLod(
        ClientWorld world,
        BlockPos playerPos,
        ObservationTransform transform,
        MappedByteBuffer buffer,
        int slotOffset,
        int lod
    ) {
        final int size = ObservationConfig.GRID_SIZE;
        final int stride = transform.getStride();
        final int subBlockCount = stride * stride * stride;
        final int playerX = playerPos.getX();
        final int playerY = playerPos.getY();
        final int playerZ = playerPos.getZ();
        final SnappedCardinal cardinal = transform.getCardinal();

        int lodBase = IpcBufferLayout.getLodBlockOffset(slotOffset, lod);

        for (int gx = 0; gx < size; gx++) {
            for (int gy = 0; gy < size; gy++) {
                int dy = transform.getDeltaY(gy);
                int originY = playerY + dy;

                for (int gz = 0; gz < size; gz++) {
                    int cellIdx = (gx * size + gy) * size + gz;
                    int dx = transform.getDeltaX(gx, gz);
                    int dz = transform.getDeltaZ(gx, gz);
                    int originX = playerX + dx;
                    int originZ = playerZ + dz;

                    Block selectedBlock;
                    VoxelPhysicalProperties props;

                    if (stride == 1) {
                        mutablePos.set(originX, originY, originZ);
                        BlockState state = world.getBlockState(mutablePos);
                        selectedBlock = state.getBlock();
                        props = PhysicalPropertyExtractor.extractSingle(state, world, mutablePos, cardinal);
                    } else {
                        cellCounts.clear();
                        accumulator.reset();
                        BlockPriority highestPriority = BlockPriority.AIR;

                        int baseLx = transform.getXLocalOffset(gx);
                        int baseLy = transform.getYLocalOffset(gy);
                        int baseLz = transform.getZLocalOffset(gz);

                        for (int ox = 0; ox < stride; ox++) {
                            int lx = baseLx + ox;
                            for (int oz = 0; oz < stride; oz++) {
                                int lz = baseLz + oz;
                                int subOriginX = playerX + cardinal.toWorldDeltaX(lx, lz);
                                int subOriginZ = playerZ + cardinal.toWorldDeltaZ(lx, lz);
                                for (int oy = 0; oy < stride; oy++) {
                                    int subOriginY = playerY + baseLy + oy;
                                    mutablePos.set(subOriginX, subOriginY, subOriginZ);
                                    BlockState state = world.getBlockState(mutablePos);
                                    Block block = state.getBlock();

                                    BlockPriority priority = BlockPriorityRegistry.getPriority(state);
                                    if (priority.isHigherPriorityThan(highestPriority)) {
                                        highestPriority = priority;
                                    }

                                    cellCounts.put(block, cellCounts.getOrDefault(block, 0) + 1);
                                    accumulator.accumulate(state, world, mutablePos);
                                }
                            }
                        }

                        selectedBlock = selectPriorityBlock(cellCounts, highestPriority);
                        props = PhysicalPropertyExtractor.aggregateSubVolume(accumulator, subBlockCount, cardinal);
                    }

                    int blockId = BlockVocabulary.getId(selectedBlock);
                    float visibility = 1.0f;

                    if (ObservationConfig.visibilityFilterEnabled && blockId != BlockVocabulary.AIR_ID) {
                        boolean exposed;
                        if (stride == 1) {
                            exposed = SurfaceShellFilter.isBlockExposed(world, mutablePos, neighborPos);
                        } else {
                            if (accumulator.solidCount < subBlockCount) {
                                exposed = true;
                            } else {
                                int baseLx = transform.getXLocalOffset(gx);
                                int baseLy = transform.getYLocalOffset(gy);
                                int baseLz = transform.getZLocalOffset(gz);
                                int minX = playerX + cardinal.toWorldDeltaX(baseLx, baseLz);
                                int minY = playerY + baseLy;
                                int minZ = playerZ + cardinal.toWorldDeltaZ(baseLx, baseLz);
                                exposed = SurfaceShellFilter.isSubVolumeExposed(world, minX, minY, minZ, stride, mutablePos, neighborPos);
                            }
                        }
                        if (!exposed) {
                            blockId = BlockVocabulary.MASKED_ID;
                            visibility = 0.0f;
                        }
                    }

                    // Direct binary writes into MappedByteBuffer
                    buffer.putShort(lodBase + IpcBufferLayout.OFF_BLOCK_ID + cellIdx * 2, (short) blockId);
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_OCCUPANCY + cellIdx * 4, props.occupancy());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_PASSABILITY + cellIdx * 4, props.passability());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_HAZARD + cellIdx * 4, props.hazard());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_HARDNESS + cellIdx * 4, props.hardness());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_LIGHT + cellIdx * 4, props.lightLevel());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_FLUID_X + cellIdx * 4, props.flowX());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_FLUID_Z + cellIdx * 4, props.flowZ());
                    buffer.putFloat(lodBase + IpcBufferLayout.OFF_VISIBILITY + cellIdx * 4, visibility);
                }
            }
        }
    }

    private void writeEntityLod(
        ClientWorld world,
        ClientPlayerEntity player,
        BlockPos playerPos,
        ObservationTransform transform,
        MappedByteBuffer buffer,
        int slotOffset,
        int lod
    ) {
        final int size = ObservationConfig.GRID_SIZE;
        final int playerX = playerPos.getX();
        final int playerY = playerPos.getY();
        final int playerZ = playerPos.getZ();

        Arrays.fill(entityCounts, 0);
        Arrays.fill(entityTypeIds, (byte) 0);
        Arrays.fill(entityHostile, 0.0f);
        Arrays.fill(entityHealth, 0.0f);

        List<Entity> entities = world.getEntitiesByClass(
            Entity.class,
            transform.getWorldBoundingBox(playerPos),
            e -> e != player && !e.isRemoved() && (!(e instanceof LivingEntity living) || living.getHealth() > 0.0f)
        );

        for (Entity entity : entities) {
            int wdx = entity.getBlockX() - playerX;
            int wdy = entity.getBlockY() - playerY;
            int wdz = entity.getBlockZ() - playerZ;

            int gx = transform.getGridX(wdx, wdz);
            int gy = transform.getGridY(wdy);
            int gz = transform.getGridZ(wdx, wdz);

            if (transform.isInGrid(gx, gy, gz)) {
                int cellIdx = (gx * size + gy) * size + gz;
                entityCounts[cellIdx]++;
                byte typeId = EntityClassifier.getNumericTypeId(entity);
                // Retain existing or prioritize hostile entity type ID if cell has multiple entities
                if (entityTypeIds[cellIdx] == 0 || (EntityClassifier.isHostile(entity) && entityHostile[cellIdx] == 0.0f)) {
                    entityTypeIds[cellIdx] = typeId;
                }
                if (EntityClassifier.isHostile(entity)) {
                    entityHostile[cellIdx] = 1.0f;
                }
                entityHealth[cellIdx] += EntityClassifier.getHealth(entity);
            }
        }

        int lodBase = IpcBufferLayout.getLodEntityOffset(slotOffset, lod);
        for (int i = 0; i < IpcBufferLayout.CELLS_PER_LOD; i++) {
            buffer.put(lodBase + IpcBufferLayout.OFF_ENTITY_COUNT + i, (byte) Math.min(255, entityCounts[i]));
            buffer.put(lodBase + IpcBufferLayout.OFF_ENTITY_TYPE_ID + i, entityTypeIds[i]);
            buffer.putFloat(lodBase + IpcBufferLayout.OFF_IS_HOSTILE + i * 4, entityHostile[i]);
            buffer.putFloat(lodBase + IpcBufferLayout.OFF_TOTAL_HEALTH + i * 4, entityHealth[i]);
        }
    }

    private Block selectPriorityBlock(Map<Block, Integer> cellCounts, BlockPriority highestPriority) {
        Block bestBlock = null;
        int bestCount = -1;

        for (Map.Entry<Block, Integer> entry : cellCounts.entrySet()) {
            Block block = entry.getKey();
            int count = entry.getValue();
            BlockPriority priority = BlockPriorityRegistry.getPriority(block);

            if (priority == highestPriority) {
                if (count > bestCount) {
                    bestCount = count;
                    bestBlock = block;
                } else if (count == bestCount) {
                    if (bestBlock == null) {
                        bestBlock = block;
                    } else {
                        String id = BlockVocabulary.getName(BlockVocabulary.getId(block));
                        String bestId = BlockVocabulary.getName(BlockVocabulary.getId(bestBlock));
                        if (id.compareTo(bestId) < 0) {
                            bestBlock = block;
                        }
                    }
                }
            }
        }

        return bestBlock != null ? bestBlock : Blocks.AIR;
    }
}

