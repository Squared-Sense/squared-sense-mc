package org.sqsense.squared_sense_mc.client;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import org.sqsense.squared_sense_mc.client.command.ModClientCommands;
import org.sqsense.squared_sense_mc.client.observation.ipc.ObservationStreamManager;
import org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary;

import java.nio.file.Path;

public class SquaredSenseMCClient implements ClientModInitializer {

    @Override
    public void onInitializeClient() {
        ModClientCommands.registerAll();
        ClientTickEvents.END_CLIENT_TICK.register(ObservationStreamManager.getInstance()::onClientTick);

        // Export canonical vocabulary to temp directory for IPC consumers
        try {
            String tempDir = System.getProperty("java.io.tmpdir");
            BlockVocabulary.exportVocabulary(Path.of(tempDir, "squared_sense_mc_vocab.json"));
        } catch (Exception ignored) {
        }
    }
}
