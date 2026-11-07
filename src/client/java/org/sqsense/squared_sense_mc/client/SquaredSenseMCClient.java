package org.sqsense.squared_sense_mc.client;

import net.fabricmc.api.ClientModInitializer;
import org.sqsense.squared_sense_mc.client.command.ModClientCommands;

public class SquaredSenseMCClient implements ClientModInitializer {

    @Override
    public void onInitializeClient() {
        ModClientCommands.registerAll();
    }
}
