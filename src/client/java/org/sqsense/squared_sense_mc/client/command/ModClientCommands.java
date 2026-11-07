package org.sqsense.squared_sense_mc.client.command;

import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;

public class ModClientCommands {
    public static void registerAll() {
        ClientCommandRegistrationCallback.EVENT.register(((dispatcher, registryAccess) -> {
            BlockInfoCommand.register(dispatcher);
            TargetBlockInfoCommand.register(dispatcher);
            SquishCommand.register(dispatcher);
            ObserveCommand.register(dispatcher);
        }));
    }
}
