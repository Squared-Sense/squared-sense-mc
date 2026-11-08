package org.sqsense.squared_sense_mc.client.observation.channels;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.fluid.FluidState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import org.sqsense.squared_sense_mc.client.observation.geometry.SnappedCardinal;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriority;
import org.sqsense.squared_sense_mc.client.observation.priority.BlockPriorityRegistry;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * High-performance extractor for the 6 physical voxel channels (Task 8):
 * 1. Occupancy / Density (solid blocks / total sub-blocks)
 * 2. Passability (1.0 air/flora, 0.5 liquid/cobweb, 0.0 solid)
 * 3. Hazard Flag (1.0 if ANY sub-block is hazard, else 0.0)
 * 4. Hardness (average break hardness across sub-blocks)
 * 5. Light Level (combined block + sky light normalized [0.0, 1.0])
 * 6. Fluid Flow (average fluid velocity rotated into egocentric frame)
 */
public final class PhysicalPropertyExtractor {
    private PhysicalPropertyExtractor() {}

    private static final Set<Block> IMPEDING_BLOCKS = new HashSet<>();
    private static final Set<Block> PASSABLE_FLORA = new HashSet<>();

    static {
        // Movement-impeding blocks (passability = 0.5)
        Collections.addAll(IMPEDING_BLOCKS,
            Blocks.COBWEB, Blocks.SWEET_BERRY_BUSH, Blocks.POWDER_SNOW,
            Blocks.VINE, Blocks.CAVE_VINES, Blocks.CAVE_VINES_PLANT,
            Blocks.TWISTING_VINES, Blocks.TWISTING_VINES_PLANT,
            Blocks.WEEPING_VINES, Blocks.WEEPING_VINES_PLANT,
            Blocks.KELP, Blocks.KELP_PLANT, Blocks.SEAGRASS, Blocks.TALL_SEAGRASS
        );

        // Fully passable flora and non-solids (passability = 1.0)
        Collections.addAll(PASSABLE_FLORA,
            Blocks.SHORT_GRASS, Blocks.TALL_GRASS, Blocks.FERN, Blocks.LARGE_FERN,
            Blocks.TORCH, Blocks.WALL_TORCH, Blocks.SOUL_TORCH, Blocks.SOUL_WALL_TORCH,
            Blocks.LANTERN, Blocks.SOUL_LANTERN, Blocks.GLOW_LICHEN, Blocks.HANGING_ROOTS,
            Blocks.SPORE_BLOSSOM, Blocks.MOSS_CARPET, Blocks.SUGAR_CANE
        );
    }

    /**
     * Determines single-block passability: 1.0 (passable), 0.5 (impeding/liquid), 0.0 (solid).
     */
    public static float getBlockPassability(BlockState state) {
        if (state.isAir()) {
            return 1.0f;
        }

        Block block = state.getBlock();
        if (IMPEDING_BLOCKS.contains(block)) {
            return 0.5f;
        }

        BlockPriority priority = BlockPriorityRegistry.getPriority(state);
        if (priority == BlockPriority.LIQUID) {
            return 0.5f;
        }

        if (priority == BlockPriority.FOLIAGE) {
            return 1.0f;
        }

        if (PASSABLE_FLORA.contains(block)) {
            return 1.0f;
        }

        return 0.0f;
    }

    /**
     * Determines whether a block is solid for occupancy calculation.
     */
    public static boolean isSolid(BlockState state) {
        if (state.isAir()) {
            return false;
        }
        if (!state.getFluidState().isEmpty()) {
            return false;
        }
        return getBlockPassability(state) == 0.0f;
    }

    /**
     * Determines whether a block triggers the hazard flag.
     */
    public static boolean isHazard(BlockState state) {
        return BlockPriorityRegistry.getPriority(state) == BlockPriority.HAZARD;
    }

    /**
     * Resolves block break hardness (0 for air, 100 for unbreakable, positive float otherwise).
     */
    public static float getHardness(BlockState state, ClientWorld world, BlockPos pos) {
        if (state.isAir()) {
            return 0.0f;
        }
        float h = state.getHardness(world, pos);
        if (h < 0.0f) {
            // Unbreakable blocks like bedrock
            return 100.0f;
        }
        return h;
    }

    /**
     * Resolves combined light level normalized to [0.0, 1.0].
     */
    public static float getNormalizedLight(ClientWorld world, BlockPos pos) {
        return world.getLightLevel(pos) / 15.0f;
    }

    /**
     * Aggregates physical channels for a single cell (LOD 0 fast-path).
     */
    public static VoxelPhysicalProperties extractSingle(
        BlockState state,
        ClientWorld world,
        BlockPos pos,
        SnappedCardinal cardinal
    ) {
        float occ = isSolid(state) ? 1.0f : 0.0f;
        float pass = getBlockPassability(state);
        float haz = isHazard(state) ? 1.0f : 0.0f;
        float hard = getHardness(state, world, pos);
        float light = getNormalizedLight(world, pos);

        float fx = 0.0f;
        float fz = 0.0f;
        FluidState fluid = state.getFluidState();
        if (!fluid.isEmpty()) {
            Vec3d vel = fluid.getVelocity(world, pos);
            fx = (float) (vel.x * cardinal.getRightX() + vel.z * cardinal.getRightZ());
            fz = (float) (vel.x * cardinal.getForwardX() + vel.z * cardinal.getForwardZ());
        }

        return new VoxelPhysicalProperties(occ, pass, haz, hard, light, fx, fz, 1.0f);
    }

    /**
     * Aggregates physical channels across all sub-blocks of a cell (LOD 1 & 2 downsampling).
     */
    public static VoxelPhysicalProperties aggregateSubVolume(
        Accumulator acc,
        int subBlockCount,
        SnappedCardinal cardinal
    ) {
        float occ = acc.solidCount / (float) subBlockCount;
        float pass = acc.passabilitySum / subBlockCount;
        float haz = acc.hasHazard ? 1.0f : 0.0f;
        float hard = acc.hardnessSum / subBlockCount;
        float light = acc.lightSum / subBlockCount;

        float fx = 0.0f;
        float fz = 0.0f;
        if (acc.fluidCount > 0) {
            float avgWorldVx = (float) (acc.worldFlowX / acc.fluidCount);
            float avgWorldVz = (float) (acc.worldFlowZ / acc.fluidCount);
            fx = avgWorldVx * cardinal.getRightX() + avgWorldVz * cardinal.getRightZ();
            fz = avgWorldVx * cardinal.getForwardX() + avgWorldVz * cardinal.getForwardZ();
        }

        return new VoxelPhysicalProperties(occ, pass, haz, hard, light, fx, fz, 1.0f);
    }

    /**
     * Reusable accumulator to prevent object allocations during sub-volume aggregation.
     */
    public static final class Accumulator {
        public int solidCount;
        public float passabilitySum;
        public boolean hasHazard;
        public float hardnessSum;
        public float lightSum;
        public int fluidCount;
        public double worldFlowX;
        public double worldFlowZ;

        public void reset() {
            solidCount = 0;
            passabilitySum = 0.0f;
            hasHazard = false;
            hardnessSum = 0.0f;
            lightSum = 0.0f;
            fluidCount = 0;
            worldFlowX = 0.0;
            worldFlowZ = 0.0;
        }

        public void accumulate(BlockState state, ClientWorld world, BlockPos pos) {
            if (isSolid(state)) {
                solidCount++;
            }
            passabilitySum += getBlockPassability(state);
            if (isHazard(state)) {
                hasHazard = true;
            }
            hardnessSum += getHardness(state, world, pos);
            lightSum += getNormalizedLight(world, pos);

            FluidState fluid = state.getFluidState();
            if (!fluid.isEmpty()) {
                Vec3d vel = fluid.getVelocity(world, pos);
                worldFlowX += vel.x;
                worldFlowZ += vel.z;
                fluidCount++;
            }
        }
    }
}

