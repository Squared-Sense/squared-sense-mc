"""
Block Palette & Static Physical Property Lookup Tables (LUTs).
Matches Java client:
- org.sqsense.squared_sense_mc.client.observation.palette.BlockVocabulary
- org.sqsense.squared_sense_mc.client.observation.priority.BlockPriority
- org.sqsense.squared_sense_mc.client.observation.priority.BlockPriorityRegistry
- org.sqsense.squared_sense_mc.client.observation.channels.PhysicalPropertyExtractor
"""

import json
import os
from typing import Dict, Optional
import numpy as np

# Priority Ranks matching BlockPriority.java
PRIORITY_AIR = 1
PRIORITY_SOLID = 2
PRIORITY_FOLIAGE = 3
PRIORITY_LIQUID = 4
PRIORITY_INTERACTIVE = 5
PRIORITY_HAZARD = 6

AIR_BLOCKS = {
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air"
}

HAZARD_BLOCKS = {
    "minecraft:lava",
    "minecraft:fire",
    "minecraft:soul_fire",
    "minecraft:campfire",
    "minecraft:soul_campfire",
    "minecraft:magma_block",
    "minecraft:cactus",
    "minecraft:sweet_berry_bush",
    "minecraft:wither_rose",
    "minecraft:powder_snow",
    "minecraft:pointed_dripstone",
    "minecraft:tnt",
    "minecraft:respawn_anchor",
    "minecraft:sculk_shrieker",
    "minecraft:sculk_sensor",
    "minecraft:calibrated_sculk_sensor"
}

LIQUID_BLOCKS = {
    "minecraft:water",
    "minecraft:bubble_column"
}

FOLIAGE_BLOCKS = {
    "minecraft:cobweb",
    "minecraft:sugar_cane",
    "minecraft:bamboo",
    "minecraft:kelp",
    "minecraft:kelp_plant",
    "minecraft:seagrass",
    "minecraft:tall_seagrass",
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:large_fern",
    "minecraft:vine",
    "minecraft:cave_vines",
    "minecraft:cave_vines_plant",
    "minecraft:twisting_vines",
    "minecraft:twisting_vines_plant",
    "minecraft:weeping_vines",
    "minecraft:weeping_vines_plant",
    "minecraft:glow_lichen",
    "minecraft:hanging_roots",
    "minecraft:spore_blossom",
    "minecraft:moss_carpet"
}

IMPEDING_BLOCKS = {
    "minecraft:cobweb",
    "minecraft:sweet_berry_bush",
    "minecraft:powder_snow",
    "minecraft:vine",
    "minecraft:cave_vines",
    "minecraft:cave_vines_plant",
    "minecraft:twisting_vines",
    "minecraft:twisting_vines_plant",
    "minecraft:weeping_vines",
    "minecraft:weeping_vines_plant",
    "minecraft:kelp",
    "minecraft:kelp_plant",
    "minecraft:seagrass",
    "minecraft:tall_seagrass"
}

PASSABLE_FLORA = {
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:large_fern",
    "minecraft:torch",
    "minecraft:wall_torch",
    "minecraft:soul_torch",
    "minecraft:soul_wall_torch",
    "minecraft:lantern",
    "minecraft:soul_lantern",
    "minecraft:glow_lichen",
    "minecraft:hanging_roots",
    "minecraft:spore_blossom",
    "minecraft:moss_carpet",
    "minecraft:sugar_cane"
}

INTERACTIVE_CONTAINERS = {
    "minecraft:chest",
    "minecraft:trapped_chest",
    "minecraft:ender_chest",
    "minecraft:barrel",
    "minecraft:hopper",
    "minecraft:dispenser",
    "minecraft:dropper",
    "minecraft:furnace",
    "minecraft:blast_furnace",
    "minecraft:smoker",
    "minecraft:crafting_table",
    "minecraft:enchanting_table",
    "minecraft:brewing_stand",
    "minecraft:anvil",
    "minecraft:chipped_anvil",
    "minecraft:damaged_anvil",
    "minecraft:smithing_table",
    "minecraft:grindstone",
    "minecraft:loom",
    "minecraft:stonecutter",
    "minecraft:cartography_table",
    "minecraft:fletching_table",
    "minecraft:beacon",
    "minecraft:conduit",
    "minecraft:lodestone",
    "minecraft:spawner",
    "minecraft:trial_spawner",
    "minecraft:vault",
    "minecraft:nether_portal",
    "minecraft:end_portal",
    "minecraft:end_portal_frame",
    "minecraft:end_gateway",
    "minecraft:lever",
    "minecraft:bell",
    "minecraft:target",
    "minecraft:lightning_rod",
    "minecraft:redstone_block",
    "minecraft:redstone_wire",
    "minecraft:repeater",
    "minecraft:comparator",
    "minecraft:redstone_torch",
    "minecraft:redstone_wall_torch",
    "minecraft:raw_iron_block",
    "minecraft:raw_copper_block",
    "minecraft:raw_gold_block",
    "minecraft:ancient_debris"
}

# Emissive light sources (luminance 1..15)
LUMINANCE_MAP = {
    "minecraft:lava": 15,
    "minecraft:fire": 15,
    "minecraft:soul_fire": 10,
    "minecraft:campfire": 15,
    "minecraft:soul_campfire": 10,
    "minecraft:beacon": 15,
    "minecraft:conduit": 15,
    "minecraft:glowstone": 15,
    "minecraft:sea_lantern": 15,
    "minecraft:shroomlight": 15,
    "minecraft:froglight": 15,
    "minecraft:ochre_froglight": 15,
    "minecraft:pearlescent_froglight": 15,
    "minecraft:verdant_froglight": 15,
    "minecraft:lantern": 15,
    "minecraft:soul_lantern": 10,
    "minecraft:torch": 14,
    "minecraft:wall_torch": 14,
    "minecraft:soul_torch": 10,
    "minecraft:soul_wall_torch": 10,
    "minecraft:end_rod": 14,
    "minecraft:crying_obsidian": 10,
    "minecraft:redstone_lamp": 15,
    "minecraft:redstone_torch": 7,
    "minecraft:redstone_wall_torch": 7,
    "minecraft:magma_block": 3,
    "minecraft:glow_lichen": 7
}

UNBREAKABLE_BLOCKS = {
    "minecraft:bedrock",
    "minecraft:barrier",
    "minecraft:command_block",
    "minecraft:repeating_command_block",
    "minecraft:chain_command_block",
    "minecraft:structure_block",
    "minecraft:jigsaw",
    "minecraft:end_portal",
    "minecraft:end_portal_frame",
    "minecraft:end_gateway"
}


def _classify_priority(name: str) -> int:
    if name in AIR_BLOCKS or name == "minecraft:air":
        return PRIORITY_AIR
    if name in HAZARD_BLOCKS or "fire" in name or "campfire" in name:
        return PRIORITY_HAZARD
    if (
        name in INTERACTIVE_CONTAINERS
        or name.endswith("_ore")
        or name.endswith("_ores")
        or "_ore_" in name
        or name.endswith("shulker_box")
        or name.endswith("_bed")
        or name.endswith("_door")
        or name.endswith("_doors")
        or name.endswith("_trapdoor")
        or name.endswith("_trapdoors")
        or name.endswith("_button")
        or name.endswith("_pressure_plate")
    ):
        return PRIORITY_INTERACTIVE
    if name in LIQUID_BLOCKS:
        return PRIORITY_LIQUID
    if (
        name in FOLIAGE_BLOCKS
        or name in PASSABLE_FLORA
        or name.endswith("_leaves")
        or name.endswith("_sapling")
        or name.endswith("_grass")
        or name == "minecraft:grass"
        or name.endswith("_fern")
        or name == "minecraft:fern"
        or name.endswith("_carpet")
        or name.endswith("_flower")
        or name.endswith("_tulip")
        or name.endswith("_orchid")
        or name.endswith("_allium")
        or name.endswith("_bluet")
        or name.endswith("_daisy")
        or name.endswith("_poppy")
        or name.endswith("_dandelion")
        or name.endswith("_rose")
        or name.endswith("_peony")
        or name.endswith("_lilac")
        or name.endswith("_bush")
        or name.endswith("_vines")
        or name.endswith("_vines_plant")
        or name == "minecraft:vine"
        or name in {
            "minecraft:wheat",
            "minecraft:carrots",
            "minecraft:potatoes",
            "minecraft:beetroots",
            "minecraft:melon_stem",
            "minecraft:pumpkin_stem",
            "minecraft:attached_melon_stem",
            "minecraft:attached_pumpkin_stem",
            "minecraft:nether_wart",
            "minecraft:cocoa",
            "minecraft:sweet_berries",
            "minecraft:torchflower_crop",
            "minecraft:pitcher_crop",
            "minecraft:pitcher_plant",
            "minecraft:dandelion",
            "minecraft:poppy",
            "minecraft:blue_orchid",
            "minecraft:allium",
            "minecraft:azure_bluet",
            "minecraft:red_tulip",
            "minecraft:orange_tulip",
            "minecraft:white_tulip",
            "minecraft:pink_tulip",
            "minecraft:oxeye_daisy",
            "minecraft:cornflower",
            "minecraft:lily_of_the_valley",
            "minecraft:sunflower",
            "minecraft:lilac",
            "minecraft:rose_bush",
            "minecraft:peony",
            "minecraft:azalea",
            "minecraft:flowering_azalea",
            "minecraft:mangrove_propagule",
            "minecraft:spore_blossom"
        }
    ):
        return PRIORITY_FOLIAGE
    return PRIORITY_SOLID


def _compute_passability(name: str, priority: int) -> float:
    if name in AIR_BLOCKS:
        return 1.0
    if name in IMPEDING_BLOCKS:
        return 0.5
    if priority == PRIORITY_LIQUID:
        return 0.5
    if priority == PRIORITY_FOLIAGE or name in PASSABLE_FLORA:
        return 1.0
    return 0.0


def _compute_hardness(name: str) -> float:
    if name in AIR_BLOCKS:
        return 0.0
    if name in UNBREAKABLE_BLOCKS:
        return 100.0
    if name == "special:masked":
        return 1.5
    if "obsidian" in name:
        return 50.0
    if "ancient_debris" in name or "anvil" in name:
        return 5.0
    if "deepslate" in name:
        return 3.0
    if "stone" in name or "cobblestone" in name or "ore" in name or "granite" in name or "diorite" in name or "andesite" in name:
        return 1.5
    if "log" in name or "wood" in name or "planks" in name:
        return 2.0
    if "dirt" in name or "sand" in name or "gravel" in name or "clay" in name:
        return 0.5
    if "leaves" in name:
        return 0.2
    if "glass" in name or "ice" in name:
        return 0.3
    if "wool" in name:
        return 0.8
    if "iron" in name or "copper" in name or "gold" in name:
        return 3.0
    return 1.0


def _compute_is_opaque(name: str, is_solid: bool) -> bool:
    if not is_solid:
        return False
    # Non-opaque solids
    non_opaque = (
        "glass" in name
        or "slab" in name
        or "stairs" in name
        or "fence" in name
        or "wall" in name
        or "door" in name
        or "trapdoor" in name
        or "ice" in name
        or "chest" in name
        or "shulker_box" in name
        or "hopper" in name
        or "anvil" in name
        or "bell" in name
        or "chain" in name
        or "lantern" in name
        or "carpet" in name
        or "leaf" in name
        or "leaves" in name
    )
    return not non_opaque


class BlockPalette:
    """
    Synchronized Block Palette with frozen integer IDs (0..N) and fast NumPy LUTs.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        if vocab_path is None:
            # Locate relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vocab_path = os.path.join(base_dir, "observations", "block_vocab.json")

        self.vocab_path = vocab_path
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data.get("vocabSize", len(data.get("idToBlock", {})))
        self.id_to_block: Dict[int, str] = {int(k): v for k, v in data.get("idToBlock", {}).items()}
        self.block_to_id: Dict[str, int] = {v: k for k, v in self.id_to_block.items()}

        # Build Look-Up Tables (LUTs) indexed by ID [0 .. vocab_size - 1]
        self.lut_priority = np.zeros(self.vocab_size, dtype=np.uint8)
        self.lut_passability = np.zeros(self.vocab_size, dtype=np.float32)
        self.lut_hazard = np.zeros(self.vocab_size, dtype=np.float32)
        self.lut_hardness = np.zeros(self.vocab_size, dtype=np.float32)
        self.lut_is_solid = np.zeros(self.vocab_size, dtype=np.bool_)
        self.lut_is_opaque = np.zeros(self.vocab_size, dtype=np.bool_)
        self.lut_luminance = np.zeros(self.vocab_size, dtype=np.uint8)

        for bid in range(self.vocab_size):
            name = self.id_to_block.get(bid, "minecraft:air")
            prio = _classify_priority(name)
            passab = _compute_passability(name, prio)
            is_haz = 1.0 if (prio == PRIORITY_HAZARD or name in HAZARD_BLOCKS) else 0.0
            hard = _compute_hardness(name)
            is_sol = (passab == 0.0 and name not in AIR_BLOCKS and name not in LIQUID_BLOCKS and name != "minecraft:lava")
            is_op = _compute_is_opaque(name, is_sol)
            lum = LUMINANCE_MAP.get(name, 0)

            self.lut_priority[bid] = prio
            self.lut_passability[bid] = passab
            self.lut_hazard[bid] = is_haz
            self.lut_hardness[bid] = hard
            self.lut_is_solid[bid] = is_sol
            self.lut_is_opaque[bid] = is_op
            self.lut_luminance[bid] = lum

        # Ensure ID 0 (air) and ID 1 (masked) invariants
        self.lut_priority[0] = PRIORITY_AIR
        self.lut_passability[0] = 1.0
        self.lut_hazard[0] = 0.0
        self.lut_hardness[0] = 0.0
        self.lut_is_solid[0] = False
        self.lut_is_opaque[0] = False

        self.lut_priority[1] = PRIORITY_SOLID
        self.lut_passability[1] = 0.0
        self.lut_hazard[1] = 0.0
        self.lut_hardness[1] = 1.5
        self.lut_is_solid[1] = True
        self.lut_is_opaque[1] = True

    def get_id(self, block_name: str) -> int:
        """Returns integer token ID for block name, defaulting to 0 (minecraft:air)."""
        clean_name = block_name.split("[")[0] if "[" in block_name else block_name
        if clean_name not in self.block_to_id and not clean_name.startswith("minecraft:"):
            clean_name = f"minecraft:{clean_name}"
        return self.block_to_id.get(clean_name, 0)

    def get_name(self, block_id: int) -> str:
        """Returns canonical block name for integer token ID."""
        return self.id_to_block.get(block_id, "minecraft:air")


_GLOBAL_PALETTE: Optional[BlockPalette] = None


def get_palette() -> BlockPalette:
    """Returns singleton BlockPalette instance."""
    global _GLOBAL_PALETTE
    if _GLOBAL_PALETTE is None:
        _GLOBAL_PALETTE = BlockPalette()
    return _GLOBAL_PALETTE
