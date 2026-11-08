package org.sqsense.squared_sense_mc.client.observation.palette;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Static canonical vocabulary mapping Minecraft blocks to frozen integer token IDs (Task 9):
 * - ID 0: minecraft:air (collapsing air, cave_air, void_air)
 * - ID 1: special:masked (unseen / occluded blocks for Phase 5)
 * - ID 2..N: All other blocks in BuiltInRegistries.BLOCK sorted alphabetically.
 *
 * Fast O(1) resolution via IdentityHashMap (pointer equality).
 * Exports to block_vocab.json for PyTorch 1:1 synchronization.
 */
public final class BlockVocabulary {
    private BlockVocabulary() {}

    public static final int AIR_ID = 0;
    public static final int MASKED_ID = 1;

    public static final String AIR_NAME = "minecraft:air";
    public static final String MASKED_NAME = "special:masked";

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final Map<Block, Integer> BLOCK_TO_ID = new IdentityHashMap<>();
    private static final Map<String, Integer> NAME_TO_ID = new HashMap<>();
    private static String[] ID_TO_NAME;
    private static Block[] ID_TO_BLOCK;
    private static int vocabSize = 0;
    private static boolean initialized = false;

    /**
     * Initializes the canonical vocabulary from the Minecraft block registry.
     * Guaranteed to be idempotent and thread-safe.
     */
    public static synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        // 1. Collect all non-air block identifiers
        List<Identifier> sortedIds = new ArrayList<>();
        for (Identifier id : Registries.BLOCK.getIds()) {
            String path = id.getPath();
            boolean isAir = id.getNamespace().equals("minecraft") &&
                (path.equals("air") || path.equals("cave_air") || path.equals("void_air"));
            if (!isAir) {
                sortedIds.add(id);
            }
        }

        // 2. Sort registry names alphabetically for deterministic freezing
        sortedIds.sort(Comparator.comparing(Identifier::toString));

        vocabSize = sortedIds.size() + 2; // 0=Air, 1=Masked, 2..N=Blocks

        ID_TO_NAME = new String[vocabSize];
        ID_TO_BLOCK = new Block[vocabSize];

        // 3. Register reserved tokens
        ID_TO_NAME[AIR_ID] = AIR_NAME;
        ID_TO_BLOCK[AIR_ID] = Blocks.AIR;

        ID_TO_NAME[MASKED_ID] = MASKED_NAME;
        ID_TO_BLOCK[MASKED_ID] = null; // Virtual token

        NAME_TO_ID.put(AIR_NAME, AIR_ID);
        NAME_TO_ID.put("minecraft:cave_air", AIR_ID);
        NAME_TO_ID.put("minecraft:void_air", AIR_ID);
        NAME_TO_ID.put(MASKED_NAME, MASKED_ID);

        BLOCK_TO_ID.put(Blocks.AIR, AIR_ID);
        BLOCK_TO_ID.put(Blocks.CAVE_AIR, AIR_ID);
        BLOCK_TO_ID.put(Blocks.VOID_AIR, AIR_ID);

        // 4. Register sorted blocks
        for (int i = 0; i < sortedIds.size(); i++) {
            int tokenIndex = i + 2;
            Identifier id = sortedIds.get(i);
            String name = id.toString();
            Block block = Registries.BLOCK.get(id);

            ID_TO_NAME[tokenIndex] = name;
            ID_TO_BLOCK[tokenIndex] = block;

            NAME_TO_ID.put(name, tokenIndex);
            BLOCK_TO_ID.put(block, tokenIndex);
        }

        initialized = true;
    }

    public static int getId(Block block) {
        if (!initialized) {
            ensureInitialized();
        }
        return BLOCK_TO_ID.getOrDefault(block, AIR_ID);
    }

    public static int getId(String name) {
        if (!initialized) {
            ensureInitialized();
        }
        return NAME_TO_ID.getOrDefault(name, AIR_ID);
    }

    public static String getName(int id) {
        if (!initialized) {
            ensureInitialized();
        }
        if (id >= 0 && id < vocabSize) {
            return ID_TO_NAME[id];
        }
        return AIR_NAME;
    }

    public static Block getBlock(int id) {
        if (!initialized) {
            ensureInitialized();
        }
        if (id >= 0 && id < vocabSize) {
            return ID_TO_BLOCK[id];
        }
        return Blocks.AIR;
    }

    public static int getVocabSize() {
        if (!initialized) {
            ensureInitialized();
        }
        return vocabSize;
    }

    /**
     * Builds and exports the block_vocab.json dictionary.
     */
    public static void exportVocabulary(Path destinationPath) throws IOException {
        if (!initialized) {
            ensureInitialized();
        }

        JsonObject root = new JsonObject();
        root.addProperty("version", 1);
        root.addProperty("vocabSize", vocabSize);

        JsonObject reserved = new JsonObject();
        reserved.addProperty("0", AIR_NAME);
        reserved.addProperty("1", MASKED_NAME);
        root.add("reservedTokens", reserved);

        JsonObject idToBlock = new JsonObject();
        for (int i = 0; i < vocabSize; i++) {
            idToBlock.addProperty(String.valueOf(i), ID_TO_NAME[i]);
        }
        root.add("idToBlock", idToBlock);

        JsonObject blockToId = new JsonObject();
        List<String> sortedNames = new ArrayList<>(NAME_TO_ID.keySet());
        Collections.sort(sortedNames);
        for (String name : sortedNames) {
            blockToId.addProperty(name, NAME_TO_ID.get(name));
        }
        root.add("blockToId", blockToId);

        if (destinationPath.getParent() != null) {
            Files.createDirectories(destinationPath.getParent());
        }
        Files.writeString(destinationPath, GSON.toJson(root), StandardCharsets.UTF_8);
    }
}
