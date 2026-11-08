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
}

