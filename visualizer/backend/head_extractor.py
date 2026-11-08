"""
Extracts official mob face textures from the Minecraft client JAR and builds
a 128x128 Entity Head Atlas along with UV coordinate mappings for Three.js rendering.
"""

import glob
import json
import os
import sys
import zipfile
from typing import Dict, Optional, Tuple
from PIL import Image

TILE_SIZE = 16
TILES_PER_ROW = 8
ATLAS_SIZE = TILE_SIZE * TILES_PER_ROW  # 128x128 pixels


def find_client_jar(custom_path: Optional[str] = None) -> Optional[str]:
    if custom_path and os.path.exists(custom_path):
        return custom_path

    home = os.path.expanduser("~")
    loom_pattern = os.path.join(
        home, ".gradle", "caches", "fabric-loom", "minecraftMaven",
        "**", "minecraft-clientonly-1.21.1*.jar"
    )
    loom_jars = glob.glob(loom_pattern, recursive=True)
    for j in loom_jars:
        if "intermediary" not in j and os.path.isfile(j):
            return j

    appdata = os.environ.get("APPDATA", "")
    minecraft_jar = os.path.join(appdata, ".minecraft", "versions", "1.21.1", "1.21.1.jar")
    if os.path.isfile(minecraft_jar):
        return minecraft_jar

    return None


def build_entity_head_atlas(jar_path: str, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    atlas_png_path = os.path.join(output_dir, "entity_heads.png")
    atlas_json_path = os.path.join(output_dir, "entity_heads.json")

    # Mob specifications: (internal_id, type_key, jar_path, crop_box, display_name, is_hostile)
    specs = [
        (1, "zombie", "assets/minecraft/textures/entity/zombie/zombie.png", (8, 8, 16, 16), "Zombie", True),
        (2, "skeleton", "assets/minecraft/textures/entity/skeleton/skeleton.png", (8, 8, 16, 16), "Skeleton", True),
        (3, "creeper", "assets/minecraft/textures/entity/creeper/creeper.png", (8, 8, 16, 16), "Creeper", True),
        (4, "spider", "assets/minecraft/textures/entity/spider/spider.png", (40, 12, 48, 20), "Spider", True),
        (5, "enderman", "assets/minecraft/textures/entity/enderman/enderman.png", (8, 8, 16, 16), "Enderman", True),
        (6, "witch", "assets/minecraft/textures/entity/witch.png", (8, 8, 16, 16), "Witch", True),
        (7, "slime", "assets/minecraft/textures/entity/slime/slime.png", (8, 8, 16, 16), "Slime", True),
        (8, "piglin", "assets/minecraft/textures/entity/piglin/piglin.png", (8, 8, 16, 16), "Piglin", True),
        (9, "blaze", "assets/minecraft/textures/entity/blaze.png", (8, 8, 16, 16), "Blaze", True),
        (10, "cow", "assets/minecraft/textures/entity/cow/cow.png", (8, 8, 16, 16), "Cow", False),
        (11, "pig", "assets/minecraft/textures/entity/pig/pig.png", (8, 8, 16, 16), "Pig", False),
        (12, "sheep", "assets/minecraft/textures/entity/sheep/sheep.png", (8, 8, 16, 16), "Sheep", False),
        (13, "chicken", "assets/minecraft/textures/entity/chicken.png", (0, 0, 16, 16), "Chicken", False),
        (14, "villager", "assets/minecraft/textures/entity/villager/villager.png", (8, 8, 16, 16), "Villager", False),
        (15, "wolf", "assets/minecraft/textures/entity/wolf/wolf.png", (2, 2, 8, 8), "Wolf", False),
        (16, "iron_golem", "assets/minecraft/textures/entity/iron_golem/iron_golem.png", (8, 8, 24, 24), "Iron Golem", False),
        (17, "player", "assets/minecraft/textures/entity/player/wide/steve.png", (8, 8, 16, 16), "Player (Steve)", False),
        (18, "polar_bear", "assets/minecraft/textures/entity/bear/polarbear.png", (9, 9, 23, 23), "Polar Bear", False),
        (19, "strider", "assets/minecraft/textures/entity/strider/strider.png", (16, 16, 32, 32), "Strider", False),
        (20, "horse", "assets/minecraft/textures/entity/horse/horse_brown.png", (6, 6, 16, 16), "Horse", False),
        (21, "llama", "assets/minecraft/textures/entity/llama/creamy.png", (6, 6, 16, 16), "Llama", False),
        (22, "cat", "assets/minecraft/textures/entity/cat/tabby.png", (5, 5, 10, 10), "Cat", False),
        (23, "fox", "assets/minecraft/textures/entity/fox/fox.png", (6, 6, 14, 14), "Fox", False),
        (24, "bat", "assets/minecraft/textures/entity/bat.png", (3, 3, 9, 9), "Bat", False),
        (25, "bee", "assets/minecraft/textures/entity/bee/bee.png", (7, 7, 14, 14), "Bee", False),
        (26, "rabbit", "assets/minecraft/textures/entity/rabbit/brown.png", (32, 8, 40, 16), "Rabbit", False),
        (27, "panda", "assets/minecraft/textures/entity/panda/panda.png", (9, 9, 22, 22), "Panda", False),
        (28, "parrot", "assets/minecraft/textures/entity/parrot/parrot_red_blue.png", (2, 2, 6, 6), "Parrot", False),
        (29, "frog", "assets/minecraft/textures/entity/frog/temperate_frog.png", (7, 7, 15, 15), "Frog", False),
        (30, "turtle", "assets/minecraft/textures/entity/turtle/big_sea_turtle.png", (6, 6, 12, 12), "Turtle", False),
        (31, "axolotl", "assets/minecraft/textures/entity/axolotl/axolotl_lucy.png", (7, 7, 15, 15), "Axolotl", False),
        (32, "allay", "assets/minecraft/textures/entity/allay/allay.png", (5, 5, 10, 10), "Allay", False),
        (33, "camel", "assets/minecraft/textures/entity/camel/camel.png", (14, 14, 28, 28), "Camel", False),
        (34, "sniffer", "assets/minecraft/textures/entity/sniffer/sniffer.png", (20, 20, 48, 48), "Sniffer", False),
        (35, "goat", "assets/minecraft/textures/entity/goat/goat.png", (34, 9, 44, 19), "Goat", False),
        (36, "armadillo", "assets/minecraft/textures/entity/armadillo.png", (6, 6, 14, 14), "Armadillo", False),
        (37, "dolphin", "assets/minecraft/textures/entity/dolphin.png", (4, 4, 12, 12), "Dolphin", False),
        (38, "squid", "assets/minecraft/textures/entity/squid/squid.png", (12, 12, 20, 20), "Squid", False),
        (39, "fish", "assets/minecraft/textures/entity/fish/salmon.png", (2, 2, 8, 8), "Fish", False),
        (40, "pillager", "assets/minecraft/textures/entity/illager/pillager.png", (8, 8, 16, 16), "Pillager", True),
        (41, "ravager", "assets/minecraft/textures/entity/illager/ravager.png", (16, 16, 32, 32), "Ravager", True),
        (42, "vex", "assets/minecraft/textures/entity/illager/vex.png", (4, 4, 12, 12), "Vex", True),
        (43, "warden", "assets/minecraft/textures/entity/warden/warden.png", (16, 16, 32, 32), "Warden", True),
        (44, "breeze", "assets/minecraft/textures/entity/breeze/breeze.png", (4, 4, 12, 12), "Breeze", True),
        (45, "ghast", "assets/minecraft/textures/entity/ghast/ghast.png", (16, 16, 32, 32), "Ghast", True),
        (46, "phantom", "assets/minecraft/textures/entity/phantom.png", (5, 5, 12, 12), "Phantom", True),
        (47, "shulker", "assets/minecraft/textures/entity/shulker/shulker.png", (16, 52, 28, 64), "Shulker", True),
        (48, "silverfish", "assets/minecraft/textures/entity/silverfish.png", (4, 4, 10, 10), "Silverfish", True),
        (49, "guardian", "assets/minecraft/textures/entity/guardian.png", (12, 12, 24, 24), "Guardian", True),
        (50, "hoglin", "assets/minecraft/textures/entity/hoglin/hoglin.png", (14, 14, 28, 28), "Hoglin", True),
        (51, "wither", "assets/minecraft/textures/entity/wither/wither.png", (8, 8, 16, 16), "Wither", True),
        (52, "ender_dragon", "assets/minecraft/textures/entity/enderdragon/dragon.png", (16, 16, 32, 32), "Ender Dragon", True),
        (53, "snow_golem", "assets/minecraft/textures/entity/snow_golem.png", (8, 8, 16, 16), "Snow Golem", False),
        (54, "hostile", "assets/minecraft/textures/entity/zombie/zombie.png", (8, 8, 16, 16), "Hostile Mob", True),
        (55, "passive", "assets/minecraft/textures/entity/cow/cow.png", (8, 8, 16, 16), "Passive Mob", False)
    ]

    aliases = {
        "minecraft:zombie": 1, "minecraft:drowned": 1, "minecraft:husk": 1, "minecraft:zombie_villager": 1, "zombie": 1,
        "minecraft:skeleton": 2, "minecraft:stray": 2, "minecraft:wither_skeleton": 2, "minecraft:bogged": 2, "skeleton": 2,
        "minecraft:creeper": 3, "creeper": 3,
        "minecraft:spider": 4, "minecraft:cave_spider": 4, "spider": 4,
        "minecraft:enderman": 5, "enderman": 5,
        "minecraft:witch": 6, "witch": 6,
        "minecraft:slime": 7, "minecraft:magma_cube": 7, "slime": 7,
        "minecraft:piglin": 8, "minecraft:piglin_brute": 8, "minecraft:zombified_piglin": 8, "piglin": 8,
        "minecraft:blaze": 9, "blaze": 9,
        "minecraft:cow": 10, "minecraft:mooshroom": 10, "cow": 10,
        "minecraft:pig": 11, "pig": 11,
        "minecraft:sheep": 12, "sheep": 12,
        "minecraft:chicken": 13, "chicken": 13,
        "minecraft:villager": 14, "minecraft:wandering_trader": 14, "villager": 14,
        "minecraft:wolf": 15, "wolf": 15,
        "minecraft:iron_golem": 16, "iron_golem": 16,
        "minecraft:player": 17, "player": 17,
        "minecraft:polar_bear": 18, "polar_bear": 18, "polarbear": 18,
        "minecraft:strider": 19, "strider": 19,
        "minecraft:horse": 20, "minecraft:donkey": 20, "minecraft:mule": 20, "minecraft:skeleton_horse": 20, "minecraft:zombie_horse": 20, "horse": 20,
        "minecraft:llama": 21, "minecraft:trader_llama": 21, "llama": 21,
        "minecraft:cat": 22, "minecraft:ocelot": 22, "cat": 22,
        "minecraft:fox": 23, "fox": 23,
        "minecraft:bat": 24, "bat": 24,
        "minecraft:bee": 25, "bee": 25,
        "minecraft:rabbit": 26, "rabbit": 26,
        "minecraft:panda": 27, "panda": 27,
        "minecraft:parrot": 28, "parrot": 28,
        "minecraft:frog": 29, "minecraft:tadpole": 29, "frog": 29,
        "minecraft:turtle": 30, "turtle": 30,
        "minecraft:axolotl": 31, "axolotl": 31,
        "minecraft:allay": 32, "allay": 32,
        "minecraft:camel": 33, "camel": 33,
        "minecraft:sniffer": 34, "sniffer": 34,
        "minecraft:goat": 35, "goat": 35,
        "minecraft:armadillo": 36, "armadillo": 36,
        "minecraft:dolphin": 37, "dolphin": 37,
        "minecraft:squid": 38, "minecraft:glow_squid": 38, "squid": 38,
        "minecraft:cod": 39, "minecraft:salmon": 39, "minecraft:pufferfish": 39, "minecraft:tropical_fish": 39, "fish": 39,
        "minecraft:pillager": 40, "minecraft:vindicator": 40, "minecraft:evoker": 40, "minecraft:illusioner": 40, "pillager": 40,
        "minecraft:ravager": 41, "ravager": 41,
        "minecraft:vex": 42, "vex": 42,
        "minecraft:warden": 43, "warden": 43,
        "minecraft:breeze": 44, "breeze": 44,
        "minecraft:ghast": 45, "ghast": 45,
        "minecraft:phantom": 46, "phantom": 46,
        "minecraft:shulker": 47, "shulker": 47,
        "minecraft:silverfish": 48, "minecraft:endermite": 48, "silverfish": 48,
        "minecraft:guardian": 49, "minecraft:elder_guardian": 49, "guardian": 49,
        "minecraft:hoglin": 50, "minecraft:zoglin": 50, "hoglin": 50,
        "minecraft:wither": 51, "wither": 51,
        "minecraft:ender_dragon": 52, "ender_dragon": 52,
        "minecraft:snow_golem": 53, "snow_golem": 53,
        "hostile": 54,
        "passive": 55
    }

    atlas_img = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))
    heads_data = {}

    with zipfile.ZipFile(jar_path, "r") as z:
        for idx, (mob_id, key, tex_path, crop_box, disp_name, is_hostile) in enumerate(specs):
            tile_x = idx % TILES_PER_ROW
            tile_y = idx // TILES_PER_ROW
            pixel_x = tile_x * TILE_SIZE
            pixel_y = tile_y * TILE_SIZE

            face_img = None
            if tex_path in z.namelist():
                try:
                    with z.open(tex_path) as f:
                        raw = Image.open(f).convert("RGBA")
                        face_img = raw.crop(crop_box).resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
                except Exception as e:
                    print(f"[HeadExtractor] Error cropping {tex_path}: {e}")

            if face_img is None:
                # Fallback color tile
                fallback_col = (255, 50, 50, 255) if is_hostile else (50, 255, 50, 255)
                face_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), fallback_col)

            atlas_img.paste(face_img, (pixel_x, pixel_y))

            # UV coordinates for Three.js (U: 0..1, V: 0..1 from bottom-left)
            u0 = tile_x / TILES_PER_ROW
            v0 = 1.0 - ((tile_y + 1) / TILES_PER_ROW)
            u1 = (tile_x + 1) / TILES_PER_ROW
            v1 = 1.0 - (tile_y / TILES_PER_ROW)

            heads_data[str(mob_id)] = {
                "id": mob_id,
                "type": key,
                "name": disp_name,
                "hostile": is_hostile,
                "tileX": tile_x,
                "tileY": tile_y,
                "u0": round(u0, 6),
                "v0": round(v0, 6),
                "u1": round(u1, 6),
                "v1": round(v1, 6)
            }

    atlas_img.save(atlas_png_path, "PNG")

    meta = {
        "version": 1,
        "tileSize": TILE_SIZE,
        "tilesPerRow": TILES_PER_ROW,
        "atlasSize": ATLAS_SIZE,
        "totalHeads": len(specs),
        "heads": heads_data,
        "aliases": aliases
    }

    with open(atlas_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[HeadExtractor] Built {atlas_png_path} ({ATLAS_SIZE}x{ATLAS_SIZE}) and {atlas_json_path}")
    return atlas_png_path, atlas_json_path


if __name__ == "__main__":
    jar = find_client_jar()
    if not jar:
        print("[HeadExtractor] Client JAR not found!")
        sys.exit(1)

    out = os.path.join(os.path.dirname(__file__), "assets")
    build_entity_head_atlas(jar, out)

