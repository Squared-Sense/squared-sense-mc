package org.sqsense.squared_sense_mc.client.observation.priority;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.registry.tag.BlockTags;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * High-speed classifier mapping Minecraft blocks to priority tiers.
 * Hierarchy: HAZARD (6) > INTERACTIVE (5) > LIQUID (4) > FOLIAGE (3) > SOLID (2) > AIR (1).
 * Results are cached in an Identity/Concurrent map for instant O(1) resolution during sampling.
 */
public final class BlockPriorityRegistry {
    private BlockPriorityRegistry() {}

    private static final Map<Block, BlockPriority> PRIORITY_CACHE = new ConcurrentHashMap<>();

    private static final Set<Block> HAZARD_BLOCKS = new HashSet<>();
    private static final Set<Block> INTERACTIVE_BLOCKS = new HashSet<>();
    private static final Set<Block> LIQUID_BLOCKS = new HashSet<>();
    private static final Set<Block> FOLIAGE_BLOCKS = new HashSet<>();
    private static final Set<Block> AIR_BLOCKS = new HashSet<>();

    static {
        // --- 1. AIR ---
        Collections.addAll(AIR_BLOCKS,
            Blocks.AIR, Blocks.CAVE_AIR, Blocks.VOID_AIR
        );

        // --- 6. HAZARDS ---
        Collections.addAll(HAZARD_BLOCKS,
            Blocks.LAVA, Blocks.FIRE, Blocks.SOUL_FIRE, Blocks.CAMPFIRE, Blocks.SOUL_CAMPFIRE,
            Blocks.MAGMA_BLOCK, Blocks.CACTUS, Blocks.SWEET_BERRY_BUSH, Blocks.WITHER_ROSE,
            Blocks.POWDER_SNOW, Blocks.POINTED_DRIPSTONE, Blocks.TNT, Blocks.RESPAWN_ANCHOR,
            Blocks.SCULK_SHRIEKER, Blocks.SCULK_SENSOR, Blocks.CALIBRATED_SCULK_SENSOR
        );

        // --- 5. INTERACTIVE / CRITICAL ---
        // Containers & Workstations
        Collections.addAll(INTERACTIVE_BLOCKS,
            Blocks.CHEST, Blocks.TRAPPED_CHEST, Blocks.ENDER_CHEST, Blocks.BARREL,
            Blocks.HOPPER, Blocks.DISPENSER, Blocks.DROPPER,
            Blocks.FURNACE, Blocks.BLAST_FURNACE, Blocks.SMOKER,
            Blocks.CRAFTING_TABLE, Blocks.ENCHANTING_TABLE, Blocks.BREWING_STAND,
            Blocks.ANVIL, Blocks.CHIPPED_ANVIL, Blocks.DAMAGED_ANVIL,
            Blocks.SMITHING_TABLE, Blocks.GRINDSTONE, Blocks.LOOM, Blocks.STONECUTTER,
            Blocks.CARTOGRAPHY_TABLE, Blocks.FLETCHING_TABLE,
            Blocks.BEACON, Blocks.CONDUIT, Blocks.LODESTONE,
            Blocks.SPAWNER, Blocks.TRIAL_SPAWNER, Blocks.VAULT,
            Blocks.NETHER_PORTAL, Blocks.END_PORTAL, Blocks.END_PORTAL_FRAME, Blocks.END_GATEWAY,
            Blocks.LEVER, Blocks.BELL, Blocks.TARGET, Blocks.LIGHTNING_ROD,
            Blocks.REDSTONE_BLOCK, Blocks.REDSTONE_WIRE, Blocks.REPEATER, Blocks.COMPARATOR,
            Blocks.REDSTONE_TORCH, Blocks.REDSTONE_WALL_TORCH
        );

        // Critical Ores
        Collections.addAll(INTERACTIVE_BLOCKS,
            Blocks.COAL_ORE, Blocks.DEEPSLATE_COAL_ORE,
            Blocks.IRON_ORE, Blocks.DEEPSLATE_IRON_ORE,
            Blocks.COPPER_ORE, Blocks.DEEPSLATE_COPPER_ORE,
            Blocks.GOLD_ORE, Blocks.DEEPSLATE_GOLD_ORE,
            Blocks.REDSTONE_ORE, Blocks.DEEPSLATE_REDSTONE_ORE,
            Blocks.LAPIS_ORE, Blocks.DEEPSLATE_LAPIS_ORE,
            Blocks.DIAMOND_ORE, Blocks.DEEPSLATE_DIAMOND_ORE,
            Blocks.EMERALD_ORE, Blocks.DEEPSLATE_EMERALD_ORE,
            Blocks.NETHER_GOLD_ORE, Blocks.NETHER_QUARTZ_ORE, Blocks.ANCIENT_DEBRIS,
            Blocks.RAW_IRON_BLOCK, Blocks.RAW_COPPER_BLOCK, Blocks.RAW_GOLD_BLOCK
        );

        // --- 4. LIQUIDS ---
        Collections.addAll(LIQUID_BLOCKS,
            Blocks.WATER, Blocks.BUBBLE_COLUMN
        );

        // --- 3. FOLIAGE / PERMEABLE ---
        Collections.addAll(FOLIAGE_BLOCKS,
            Blocks.COBWEB, Blocks.SUGAR_CANE, Blocks.BAMBOO, Blocks.KELP, Blocks.KELP_PLANT,
            Blocks.SEAGRASS, Blocks.TALL_SEAGRASS, Blocks.SHORT_GRASS, Blocks.TALL_GRASS,
            Blocks.FERN, Blocks.LARGE_FERN, Blocks.VINE, Blocks.CAVE_VINES, Blocks.CAVE_VINES_PLANT,
            Blocks.TWISTING_VINES, Blocks.TWISTING_VINES_PLANT, Blocks.WEEPING_VINES,
            Blocks.WEEPING_VINES_PLANT, Blocks.GLOW_LICHEN, Blocks.HANGING_ROOTS,
            Blocks.SPORE_BLOSSOM, Blocks.MOSS_CARPET
        );
    }

    /**
     * Resolves the priority tier of a block state using cached lookups and tag fallbacks.
     */
    public static BlockPriority getPriority(BlockState state) {
        Block block = state.getBlock();
        BlockPriority cached = PRIORITY_CACHE.get(block);
        if (cached != null) {
            return cached;
        }

        BlockPriority resolved = classify(state, block);
        PRIORITY_CACHE.put(block, resolved);
        return resolved;
    }

    public static BlockPriority getPriority(Block block) {
        return getPriority(block.getDefaultState());
    }

    private static BlockPriority classify(BlockState state, Block block) {
        if (state.isAir() || AIR_BLOCKS.contains(block)) {
            return BlockPriority.AIR;
        }

        if (HAZARD_BLOCKS.contains(block) || state.isIn(BlockTags.FIRE) || state.isIn(BlockTags.CAMPFIRES)) {
            return BlockPriority.HAZARD;
        }

        if (INTERACTIVE_BLOCKS.contains(block)
                || isOre(state)
                || state.isIn(BlockTags.SHULKER_BOXES)
                || state.isIn(BlockTags.BEDS)
                || state.isIn(BlockTags.DOORS)
                || state.isIn(BlockTags.TRAPDOORS)
                || state.isIn(BlockTags.BUTTONS)) {
            return BlockPriority.INTERACTIVE;
        }

        if (LIQUID_BLOCKS.contains(block)) {
            return BlockPriority.LIQUID;
        }

        if (FOLIAGE_BLOCKS.contains(block)
                || state.isIn(BlockTags.LEAVES)
                || state.isIn(BlockTags.FLOWERS)
                || state.isIn(BlockTags.CROPS)
                || state.isIn(BlockTags.SAPLINGS)) {
            return BlockPriority.FOLIAGE;
        }

        return BlockPriority.SOLID;
    }

    private static boolean isOre(BlockState state) {
        return state.isIn(BlockTags.COAL_ORES)
                || state.isIn(BlockTags.IRON_ORES)
                || state.isIn(BlockTags.COPPER_ORES)
                || state.isIn(BlockTags.GOLD_ORES)
                || state.isIn(BlockTags.REDSTONE_ORES)
                || state.isIn(BlockTags.LAPIS_ORES)
                || state.isIn(BlockTags.DIAMOND_ORES)
                || state.isIn(BlockTags.EMERALD_ORES);
    }
}

