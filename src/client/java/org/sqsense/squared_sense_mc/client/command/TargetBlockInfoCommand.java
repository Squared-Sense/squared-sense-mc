package org.sqsense.squared_sense_mc.client.command;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.minecraft.text.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sqsense.squared_sense_mc.client.util.BlockInspector;

public class TargetBlockInfoCommand {
    private static final Logger LOGGER = LoggerFactory.getLogger("squared_sense_mc");

    public static void register(CommandDispatcher<FabricClientCommandSource> dispatcher) {
        dispatcher.register(ClientCommandManager.literal("sense_target").executes(TargetBlockInfoCommand::execute));
    }

    public static int execute(CommandContext<FabricClientCommandSource> context) {
        var client = context.getSource().getClient();
        var targetOpt = BlockInspector.getBlockFromTarget(client);

        if (targetOpt.isEmpty()) {
            context.getSource().sendError(Text.literal("Air"));
            return 0;
        }

        var target = targetOpt.get();
        LOGGER.info("Target Block: {} at {}", target.id(), target.pos().toShortString());

        context.getSource().sendFeedback(
                Text.literal("§a[BlockInfo]§r " + target.id() + " at " + target.pos().toShortString())
        );

        return Command.SINGLE_SUCCESS;
    }
}
