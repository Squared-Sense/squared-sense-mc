package org.sqsense.squared_sense_mc.client.command;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.command.argument.BlockPosArgumentType;
import net.minecraft.command.argument.DefaultPosArgument;
import net.minecraft.registry.Registries;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.block.BlockState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BlockInfoCommand {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");

    public static void register(CommandDispatcher<FabricClientCommandSource> dispatcher) {
        dispatcher.register(
                ClientCommandManager.literal("sense")
                        .then(ClientCommandManager.argument("pos", BlockPosArgumentType.blockPos())
                                .executes(BlockInfoCommand::execute)
                        )
        );
    }

    public static int execute(CommandContext<FabricClientCommandSource> context) {
        var client = context.getSource().getClient();
        var world = client.world;

        if (world == null) {
            context.getSource().sendError(Text.literal("World not loaded."));
            return 0;
        }

        DefaultPosArgument posArgument = context.getArgument("pos", DefaultPosArgument.class);

        Vec3d sourcePos = context.getSource().getPosition(); // Resolves ~ relative inputs correctly
        BlockPos targetPos = posArgument.toAbsoluteBlockPos(new net.minecraft.server.command.ServerCommandSource(
                null, sourcePos, null, null, 0, null, null, null, null
        ));

        BlockState state = world.getBlockState(targetPos);
        var blockId = Registries.BLOCK.getId(state.getBlock()).toString();

        if (state.isAir()) {
            context.getSource().sendError(Text.literal("Air at " + targetPos.toShortString()));
            return 0;
        }

        LOGGER.info("Block at {}: {}", targetPos.toShortString(), blockId);

        context.getSource().sendFeedback(
                Text.literal("§a[BlockInfo]§r " + blockId + " at " + targetPos.toShortString())
        );

        return Command.SINGLE_SUCCESS;
    }
}
