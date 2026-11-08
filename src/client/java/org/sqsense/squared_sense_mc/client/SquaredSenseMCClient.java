package org.sqsense.squared_sense_mc.client;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import org.sqsense.squared_sense_mc.client.command.ModClientCommands;
import org.sqsense.squared_sense_mc.client.observation.ipc.ObservationStreamManager;

public class SquaredSenseMCClient implements ClientModInitializer {

    @Override
    public void onInitializeClient() {
        ModClientCommands.registerAll();
        ClientTickEvents.END_CLIENT_TICK.register(ObservationStreamManager.getInstance()::onClientTick);
    }
}
