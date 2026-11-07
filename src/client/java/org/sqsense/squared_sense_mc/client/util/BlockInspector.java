package org.sqsense.squared_sense_mc.client.util;

import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.registry.Registries;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import java.util.Optional;

public final class BlockInspector {
    private BlockInspector() {}

    public record BlockTarget(BlockPos pos, BlockState state, String id) {}

    public static Optional<BlockTarget> getBlockFromTarget(MinecraftClient client) {
        HitResult hit = client.crosshairTarget;
        if (hit != null && hit.getType() == HitResult.Type.BLOCK && client.world != null) {
            BlockPos pos = ((BlockHitResult) hit).getBlockPos();
            BlockState state = client.world.getBlockState(pos);
            String id = Registries.BLOCK.getId(state.getBlock()).toString();

            return Optional.of(new BlockTarget(pos, state, id));
        }

        return Optional.empty();
    }

    public static Optional<BlockTarget> getBlockFromPos(MinecraftClient client, BlockPos pos) {
        if (client.world != null) {
            BlockState state = client.world.getBlockState(pos);
            String id = Registries.BLOCK.getId(state.getBlock()).toString();

            return Optional.of(new BlockTarget(pos, state, id));
        }

        return Optional.empty();
    }
}
