package org.sqsense.squared_sense_mc.client.observation.visibility;

import net.minecraft.block.BlockState;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;

/**
 * High-performance Exposed-Face Surface Shell Filter for Phase 5 Visibility & Occlusion Masking.
 *
 * Checks if a non-air block has at least one exposed face touching air or non-opaque blocks.
 * If a block is completely surrounded on all 6 sides by opaque full cubes, it is classified
 * as occluded underworld terrain and assigned the MASKED token (ID 1) with visibility 0.0.
 */
public final class SurfaceShellFilter {
    private SurfaceShellFilter() {}

    private static final Direction[] DIRECTIONS = Direction.values();

    /**
     * Determines whether the block at {@code pos} is exposed to open space or a non-opaque block.
     * Uses the provided {@code neighborPos} mutable buffer to ensure zero object allocations.
     *
     * @return true if at least one face touches air or a non-opaque block; false if completely enclosed.
     */
    public static boolean isBlockExposed(ClientWorld world, BlockPos pos, BlockPos.Mutable neighborPos) {
        int x = pos.getX();
        int y = pos.getY();
        int z = pos.getZ();

        for (Direction dir : DIRECTIONS) {
            neighborPos.set(x + dir.getOffsetX(), y + dir.getOffsetY(), z + dir.getOffsetZ());
            BlockState neighborState = world.getBlockState(neighborPos);
            if (!neighborState.isOpaqueFullCube(world, neighborPos)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Determines whether a multi-block sub-volume (LOD 1 or LOD 2) has any exposed outer surface faces.
     * Samples the 6 outer bounding faces of the {@code stride x stride x stride} cube.
     *
     * @return true if any bounding neighbor is non-opaque; false if completely surrounded by opaque blocks.
     */
    public static boolean isSubVolumeExposed(
        ClientWorld world,
        int minX, int minY, int minZ,
        int stride,
        BlockPos.Mutable mutablePos,
        BlockPos.Mutable neighborPos
    ) {
        int maxX = minX + stride - 1;
        int maxY = minY + stride - 1;
        int maxZ = minZ + stride - 1;

        // 1. Top (+Y) and Bottom (-Y) faces
        for (int ox = 0; ox < stride; ox++) {
            for (int oz = 0; oz < stride; oz++) {
                neighborPos.set(minX + ox, maxY + 1, minZ + oz);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
                neighborPos.set(minX + ox, minY - 1, minZ + oz);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
            }
        }

        // 2. Lateral faces (East +X, West -X, South +Z, North -Z)
        for (int oy = 0; oy < stride; oy++) {
            for (int oz = 0; oz < stride; oz++) {
                neighborPos.set(maxX + 1, minY + oy, minZ + oz);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
                neighborPos.set(minX - 1, minY + oy, minZ + oz);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
            }
            for (int ox = 0; ox < stride; ox++) {
                neighborPos.set(minX + ox, minY + oy, maxZ + 1);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
                neighborPos.set(minX + ox, minY + oy, minZ - 1);
                if (!world.getBlockState(neighborPos).isOpaqueFullCube(world, neighborPos)) {
                    return true;
                }
            }
        }

        return false;
    }
}

