package org.sqsense.squared_sense_mc.client.observation.sampler;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.util.math.BlockPos;
import org.sqsense.squared_sense_mc.client.observation.ObservationConfig;
import org.sqsense.squared_sense_mc.client.observation.channels.PhysicalPropertyExtractor;
import org.sqsense.squared_sense_mc.client.observation.channels.VoxelPhysicalProperties;
import org.sqsense.squared_sense_mc.client.observation.geometry.ObservationTransform;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriority;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriorityRegistry;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * High-performance block state, integer token ID, and physical channel sampler for fixed 16x16x16 grids:
 * - Task 7: Priority Downsampling (Hazards > Interactive > Liquids > Foliage > Solid > Air).
 * - Task 8: Multi-channel physical feature extraction (occupancy, passability, hazard, hardness, light, fluid flow).
 * - Task 9: Integer Palette Mapping (frozen canonical integer IDs 0..N, with 0=Air, 1=Masked).
 */
public class BlockSampler {

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

                    Block selectedBlock;
                    VoxelPhysicalProperties physicalProperties;

                    if (stride == 1) {
                        // Fast-path LOD 0: 1:1 direct lookup
                        mutablePos.set(originX, originY, originZ);
                        BlockState state = world.getBlockState(mutablePos);
                        selectedBlock = state.getBlock();
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

                        int baseLx = transform.getXLocalOffset(gx);
                        int baseLy = transform.getYLocalOffset(gy);
                        int baseLz = transform.getZLocalOffset(gz);
                        SnappedCardinal cardinal = transform.getCardinal();

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
                        physicalProperties = PhysicalPropertyExtractor.aggregateSubVolume(
                            accumulator,
                            subBlockCount,
                            transform.getCardinal()
                        );
                    }

                    int blockId = BlockVocabulary.getId(selectedBlock);
                    String blockName = BlockVocabulary.getName(blockId);

                    if (blockId == BlockVocabulary.AIR_ID) {
                        airCells++;
                    }
                    globalCounts.put(blockName, globalCounts.getOrDefault(blockName, 0) + 1);

                    JsonObject cellObj = new JsonObject();
                    cellObj.addProperty("id", blockId);
                    cellObj.addProperty("name", blockName);
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

    private static Block selectPriorityBlock(Map<Block, Integer> cellCounts, BlockPriority highestPriority) {
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

        if (bestBlock == null) {
            bestBlock = Blocks.AIR;
        }

        return bestBlock;
    }
}
