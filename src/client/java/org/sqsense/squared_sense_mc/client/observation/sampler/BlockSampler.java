package org.sqsense.squared_sense_mc.client.observation.sampler;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.registry.Registries;
import net.minecraft.util.math.BlockPos;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.channels.PhysicalPropertyExtractor;
import org.sqsense.squared_sense_mc.client.observation.channels.VoxelPhysicalProperties;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriority;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriorityRegistry;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * High-performance block state and physical channel sampler for fixed 16x16x16 grids:
 * - Uses single BlockPos.Mutable to eliminate thousands of coordinate allocations.
 * - Reuses single cell frequency map and single accumulator across cells.
 * - Caches Block -> String registry IDs.
 * - Task 7: Priority Downsampling (Hazards > Interactive > Liquids > Foliage > Solid > Air).
 * - Task 8: Multi-channel physical feature extraction (occupancy, passability, hazard, hardness, light, fluid flow).
 */
public class BlockSampler {
    private static final Map<Block, String> BLOCK_ID_CACHE = new ConcurrentHashMap<>();

    public static SamplingResult sample(
        ClientWorld world,
        BlockPos playerPos,
        ObservationTransform transform
    ) {
        final int size = ObservationConfig.GRID_SIZE;
        final int stride = transform.getStride();
        final int subBlockCount = stride * stride * stride;
        final int playerX = playerPos.getX();
        final int playerY = playerPos.getY();
        final int playerZ = playerPos.getZ();

        BlockPos.Mutable mutablePos = new BlockPos.Mutable();
        Map<Block, Integer> cellCounts = new HashMap<>(32);
        PhysicalPropertyExtractor.Accumulator accumulator = new PhysicalPropertyExtractor.Accumulator();
        Map<String, Integer> globalCounts = new LinkedHashMap<>();

        JsonArray xArray = new JsonArray();
        int airCells = 0;

        for (int gx = 0; gx < size; gx++) {
            JsonArray yArray = new JsonArray();
            for (int gy = 0; gy < size; gy++) {
                int dy = transform.getDeltaY(gy);
                int originY = playerY + dy;

                JsonArray zArray = new JsonArray();
                for (int gz = 0; gz < size; gz++) {
                    int dx = transform.getDeltaX(gx, gz);
                    int dz = transform.getDeltaZ(gx, gz);
                    int originX = playerX + dx;
                    int originZ = playerZ + dz;

                    String selectedBlock;
                    VoxelPhysicalProperties physicalProperties;

                    if (stride == 1) {
                        // Fast-path LOD 0: 1:1 direct lookup
                        mutablePos.set(originX, originY, originZ);
                        BlockState state = world.getBlockState(mutablePos);
                        selectedBlock = getBlockId(state.getBlock());
                        physicalProperties = PhysicalPropertyExtractor.extractSingle(
                            state,
                            world,
                            mutablePos,
                            transform.getCardinal()
                        );
                    } else {
                        // LOD downsampling with Task 7 Priority Hierarchy & Task 8 Property Aggregation
                        cellCounts.clear();
                        accumulator.reset();
                        BlockPriority highestPriority = BlockPriority.AIR;

                        for (int ox = 0; ox < stride; ox++) {
                            for (int oy = 0; oy < stride; oy++) {
                                for (int oz = 0; oz < stride; oz++) {
                                    mutablePos.set(originX + ox, originY + oy, originZ + oz);
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
                        physicalProperties = PhysicalPropertyExtractor.aggregateSubVolume(
                            accumulator,
                            subBlockCount,
                            transform.getCardinal()
                        );
                    }

                    if (isAirId(selectedBlock)) {
                        airCells++;
                    }
                    globalCounts.put(selectedBlock, globalCounts.getOrDefault(selectedBlock, 0) + 1);

                    JsonObject cellObj = new JsonObject();
                    cellObj.addProperty("name", selectedBlock);
                    cellObj.addProperty("dx", dx);
                    cellObj.addProperty("dy", dy);
                    cellObj.addProperty("dz", dz);
                    cellObj.add("properties", physicalProperties.toJsonObject());

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

    private static String selectPriorityBlock(Map<Block, Integer> cellCounts, BlockPriority highestPriority) {
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
                        String id = getBlockId(block);
                        String bestId = getBlockId(bestBlock);
                        if (id.compareTo(bestId) < 0) {
                            bestBlock = block;
                        }
                    }
                }
            }
        }

        if (bestBlock == null) {
            bestBlock = Blocks.AIR;
        }

        return getBlockId(bestBlock);
    }

    public static String getBlockId(Block block) {
        return BLOCK_ID_CACHE.computeIfAbsent(block, b -> Registries.BLOCK.getId(b).toString());
    }

    public static boolean isAirId(String id) {
        return "minecraft:air".equals(id) || "minecraft:cave_air".equals(id) || "minecraft:void_air".equals(id);
    }
}
