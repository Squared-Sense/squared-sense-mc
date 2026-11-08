package org.sqsense.squared_sense_mc.client.observation.entity;

import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.Angerable;
import net.minecraft.entity.mob.Monster;
import net.minecraft.registry.Registries;

/**
 * Evaluates entity classifications (hostility, health, registry names) for RL observation grids (Task 10).
 */
public final class EntityClassifier {
    private EntityClassifier() {}

    /**
     * Determines whether an entity is hostile towards the player:
     * - Inherently hostile: any entity implementing net.minecraft.entity.mob.Monster.
     * - Neutral/angerable: Angerable mobs with active anger timer > 0.
     */
    public static boolean isHostile(Entity entity) {
        if (entity instanceof Monster) {
            return true;
        }
        if (entity instanceof Angerable angerable) {
            return angerable.hasAngerTime();
        }
        return false;
    }

    /**
     * Retrieves the current health of an entity.
     * Non-living entities (e.g. minecarts, items, projectiles) return 0.0f.
     */
    public static float getHealth(Entity entity) {
        if (entity instanceof LivingEntity living) {
            return Math.max(0.0f, living.getHealth());
        }
        return 0.0f;
    }

    /**
     * Returns the canonical registry identifier for the entity type (e.g. "minecraft:zombie").
     */
    public static String getTypeId(Entity entity) {
        return Registries.ENTITY_TYPE.getId(entity.getType()).toString();
    }

    /**
     * Maps canonical entity type to a compact byte token ID for IPC and 3D visualizer mob heads.
     */
    public static byte getNumericTypeId(Entity entity) {
        String id = getTypeId(entity);
        return switch (id) {
            case "minecraft:zombie", "minecraft:drowned", "minecraft:husk", "minecraft:zombie_villager" -> 1;
            case "minecraft:skeleton", "minecraft:stray", "minecraft:wither_skeleton", "minecraft:bogged" -> 2;
            case "minecraft:creeper" -> 3;
            case "minecraft:spider", "minecraft:cave_spider" -> 4;
            case "minecraft:enderman" -> 5;
            case "minecraft:witch" -> 6;
            case "minecraft:slime", "minecraft:magma_cube" -> 7;
            case "minecraft:piglin", "minecraft:piglin_brute", "minecraft:zombified_piglin" -> 8;
            case "minecraft:blaze" -> 9;
            case "minecraft:cow", "minecraft:mooshroom" -> 10;
            case "minecraft:pig" -> 11;
            case "minecraft:sheep" -> 12;
            case "minecraft:chicken" -> 13;
            case "minecraft:villager", "minecraft:wandering_trader" -> 14;
            case "minecraft:wolf" -> 15;
            case "minecraft:iron_golem" -> 16;
            case "minecraft:player" -> 17;
            default -> isHostile(entity) ? (byte) 18 : (byte) 19;
        };
    }
}

