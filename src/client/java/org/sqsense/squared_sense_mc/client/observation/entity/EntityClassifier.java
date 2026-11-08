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
        String id = getTypeId(entity);
        return switch (id) {
            case "minecraft:tnt",
                 "minecraft:fireball",
                 "minecraft:small_fireball",
                 "minecraft:dragon_fireball",
                 "minecraft:wither_skull",
                 "minecraft:shulker_bullet",
                 "minecraft:wind_charge",
                 "minecraft:breeze_wind_charge",
                 "minecraft:arrow",
                 "minecraft:spectral_arrow",
                 "minecraft:trident" -> true;
            default -> false;
        };
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
        return (byte) switch (id) {
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
            case "minecraft:polar_bear" -> 18;
            case "minecraft:strider" -> 19;
            case "minecraft:horse", "minecraft:donkey", "minecraft:mule", "minecraft:skeleton_horse", "minecraft:zombie_horse" -> 20;
            case "minecraft:llama", "minecraft:trader_llama" -> 21;
            case "minecraft:cat", "minecraft:ocelot" -> 22;
            case "minecraft:fox" -> 23;
            case "minecraft:bat" -> 24;
            case "minecraft:bee" -> 25;
            case "minecraft:rabbit" -> 26;
            case "minecraft:panda" -> 27;
            case "minecraft:parrot" -> 28;
            case "minecraft:frog", "minecraft:tadpole" -> 29;
            case "minecraft:turtle" -> 30;
            case "minecraft:axolotl" -> 31;
            case "minecraft:allay" -> 32;
            case "minecraft:camel" -> 33;
            case "minecraft:sniffer" -> 34;
            case "minecraft:goat" -> 35;
            case "minecraft:armadillo" -> 36;
            case "minecraft:dolphin" -> 37;
            case "minecraft:squid", "minecraft:glow_squid" -> 38;
            case "minecraft:cod", "minecraft:salmon", "minecraft:pufferfish", "minecraft:tropical_fish" -> 39;
            case "minecraft:pillager", "minecraft:vindicator", "minecraft:evoker", "minecraft:illusioner" -> 40;
            case "minecraft:ravager" -> 41;
            case "minecraft:vex" -> 42;
            case "minecraft:warden" -> 43;
            case "minecraft:breeze" -> 44;
            case "minecraft:ghast" -> 45;
            case "minecraft:phantom" -> 46;
            case "minecraft:shulker" -> 47;
            case "minecraft:silverfish", "minecraft:endermite" -> 48;
            case "minecraft:guardian", "minecraft:elder_guardian" -> 49;
            case "minecraft:hoglin", "minecraft:zoglin" -> 50;
            case "minecraft:wither" -> 51;
            case "minecraft:ender_dragon" -> 52;
            case "minecraft:snow_golem" -> 53;
            case "minecraft:arrow", "minecraft:spectral_arrow" -> 56;
            case "minecraft:trident" -> 57;
            case "minecraft:fireball", "minecraft:small_fireball", "minecraft:dragon_fireball" -> 58;
            case "minecraft:wind_charge", "minecraft:breeze_wind_charge" -> 59;
            case "minecraft:wither_skull" -> 60;
            case "minecraft:shulker_bullet" -> 61;
            case "minecraft:snowball", "minecraft:egg" -> 62;
            case "minecraft:ender_pearl", "minecraft:eye_of_ender" -> 63;
            case "minecraft:potion", "minecraft:splash_potion", "minecraft:lingering_potion", "minecraft:experience_bottle" -> 64;
            case "minecraft:tnt" -> 65;
            case "minecraft:end_crystal" -> 66;
            case "minecraft:item" -> 67;
            case "minecraft:experience_orb" -> 68;
            case "minecraft:minecart", "minecraft:chest_minecart", "minecraft:furnace_minecart",
                 "minecraft:tnt_minecart", "minecraft:hopper_minecart", "minecraft:spawner_minecart",
                 "minecraft:command_block_minecart" -> 69;
            case "minecraft:boat", "minecraft:chest_boat" -> 70;
            case "minecraft:armor_stand" -> 71;
            case "minecraft:falling_block" -> 72;
            default -> isHostile(entity) ? (byte) 54 : (byte) 55;
        };
    }
}

