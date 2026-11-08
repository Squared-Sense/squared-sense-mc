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
        (18, "hostile", "assets/minecraft/textures/entity/zombie/zombie.png", (8, 8, 16, 16), "Hostile Mob", True),
        (19, "passive", "assets/minecraft/textures/entity/cow/cow.png", (8, 8, 16, 16), "Passive Mob", False)
    ]

    aliases = {
        "minecraft:zombie": 1,
        "minecraft:drowned": 1,
        "minecraft:husk": 1,
        "minecraft:zombie_villager": 1,
        "zombie": 1,
        "minecraft:skeleton": 2,
        "minecraft:stray": 2,
        "minecraft:wither_skeleton": 2,
        "minecraft:bogged": 2,
        "skeleton": 2,
        "minecraft:creeper": 3,
        "creeper": 3,
        "minecraft:spider": 4,
        "minecraft:cave_spider": 4,
        "spider": 4,
        "minecraft:enderman": 5,
        "enderman": 5,
        "minecraft:witch": 6,
        "witch": 6,
        "minecraft:slime": 7,
        "minecraft:magma_cube": 7,
        "slime": 7,
        "minecraft:piglin": 8,
        "minecraft:piglin_brute": 8,
        "minecraft:zombified_piglin": 8,
        "piglin": 8,
        "minecraft:blaze": 9,
        "blaze": 9,
        "minecraft:cow": 10,
        "minecraft:mooshroom": 10,
        "cow": 10,
        "minecraft:pig": 11,
        "pig": 11,
        "minecraft:sheep": 12,
        "sheep": 12,
        "minecraft:chicken": 13,
        "chicken": 13,
        "minecraft:villager": 14,
        "minecraft:wandering_trader": 14,
        "villager": 14,
        "minecraft:wolf": 15,
        "wolf": 15,
        "minecraft:iron_golem": 16,
        "iron_golem": 16,
        "minecraft:player": 17,
        "player": 17,
        "hostile": 18,
        "passive": 19
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

