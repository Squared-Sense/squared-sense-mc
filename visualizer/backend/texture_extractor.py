"""
Extracts official 16x16 block textures from the Minecraft client JAR and builds
a 512x512 Texture Atlas along with UV coordinate mappings for Three.js instanced rendering.
"""

import glob
import json
import os
import sys
import zipfile
from typing import Dict, List, Optional, Tuple
from PIL import Image

TILE_SIZE = 16
TILES_PER_ROW = 32  # 32x32 tiles = 1024 total slots
ATLAS_SIZE = TILE_SIZE * TILES_PER_ROW  # 512x512 pixels


def find_client_jar(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the Minecraft client JAR in gradle loom cache or user .minecraft directory."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Search in Loom gradle cache
    home = os.path.expanduser("~")
    loom_pattern = os.path.join(
        home, ".gradle", "caches", "fabric-loom", "minecraftMaven",
        "**", "minecraft-clientonly-1.21.1*.jar"
    )
    loom_jars = glob.glob(loom_pattern, recursive=True)
    for j in loom_jars:
        if "intermediary" not in j and os.path.isfile(j):
            return j

    # Search in standard .minecraft directory
    appdata = os.environ.get("APPDATA", "")
    minecraft_jar = os.path.join(appdata, ".minecraft", "versions", "1.21.1", "1.21.1.jar")
    if os.path.isfile(minecraft_jar):
        return minecraft_jar

    return None


def find_vocab_json(custom_path: Optional[str] = None) -> Optional[str]:
    """Locates observations/block_vocab.json."""
    candidates = [
        custom_path,
        os.path.join(os.path.dirname(__file__), "..", "..", "observations", "block_vocab.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "run", "observations", "block_vocab.json"),
        "observations/block_vocab.json"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def get_preferred_texture_name(block_name: str, available_textures: Dict[str, str]) -> Optional[str]:
    """
    Finds the best matching texture file in available_textures for a block name.
    Example: 'minecraft:grass_block' -> 'grass_block_top.png'
    """
    path = block_name.split(":")[-1]

    # Preference order for block faces
    candidates = [
        f"{path}.png",
        f"{path}_top.png",
        f"{path}_front.png",
        f"{path}_side.png",
        f"{path}_bottom.png",
        f"{path}_still.png",
        f"{path}_0.png",
        f"{path}_stage0.png",
        f"{path}_1.png",
        f"{path}_open.png"
    ]

    for c in candidates:
        if c in available_textures:
            return available_textures[c]

    # Prefix / substring search fallback
    for name, full_path in available_textures.items():
        if name.startswith(path):
            return full_path

    return None


def build_texture_atlas(
    jar_path: str,
    output_dir: str,
    vocab_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Extracts block textures from client JAR, stitches into a single 512x512 atlas,
    and produces atlas_uvs.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    atlas_png_path = os.path.join(output_dir, "texture_atlas.png")
    uv_json_path = os.path.join(output_dir, "atlas_uvs.json")

    print(f"[TextureExtractor] Reading client JAR: {jar_path}")
    zip_file = zipfile.ZipFile(jar_path, "r")

    # 1. Index all block textures in the JAR
    available_textures: Dict[str, str] = {}
    for filename in zip_file.namelist():
        if filename.startswith("assets/minecraft/textures/block/") and filename.endswith(".png"):
            basename = os.path.basename(filename)
            available_textures[basename] = filename

    print(f"[TextureExtractor] Discovered {len(available_textures)} block textures in JAR.")

    # 2. Determine vocabulary list
    vocab_file = find_vocab_json(vocab_path)
    id_to_name: Dict[int, str] = {}
    if vocab_file:
        print(f"[TextureExtractor] Using canonical block_vocab.json: {vocab_file}")
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            id_to_name = {int(k): v for k, v in data.get("idToBlock", {}).items()}
    else:
        print("[TextureExtractor] block_vocab.json not found on disk yet. Constructing alphabetized fallback...")
        # Fallback: Alphabetize all texture base names as vanilla blocks
        id_to_name[0] = "minecraft:air"
        id_to_name[1] = "special:masked"
        sorted_bases = sorted(set(os.path.splitext(b)[0] for b in available_textures.keys()))
        for i, base in enumerate(sorted_bases):
            id_to_name[i + 2] = f"minecraft:{base}"

    # 3. Create 512x512 RGBA Atlas Canvas
    atlas_img = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))

    # Tile 0: Air (fully transparent, already initialized)
    # Tile 1: Masked (translucent dark crosshatch)
    masked_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (30, 30, 35, 200))
    atlas_img.paste(masked_img, (TILE_SIZE, 0))  # tileX=1, tileY=0

    uv_mapping: Dict[str, Dict] = {}

    # Reserve tile 0 and 1
    uv_mapping["0"] = {
        "id": 0,
        "name": "minecraft:air",
        "tileX": 0,
        "tileY": 0,
        "u0": 0.0,
        "v0": 1.0 - (1.0 / TILES_PER_ROW),
        "u1": 1.0 / TILES_PER_ROW,
        "v1": 1.0,
        "avgColor": "#000000"
    }

    uv_mapping["1"] = {
        "id": 1,
        "name": "special:masked",
        "tileX": 1,
        "tileY": 0,
        "u0": 1.0 / TILES_PER_ROW,
        "v0": 1.0 - (1.0 / TILES_PER_ROW),
        "u1": 2.0 / TILES_PER_ROW,
        "v1": 1.0,
        "avgColor": "#1e1e23"
    }

    # 4. Map and paste each block texture
    next_slot = 2
    matched_count = 0
    fallback_color_count = 0

    # Sort by token ID
    sorted_tokens = sorted(id_to_name.items())

    for token_id, block_name in sorted_tokens:
        if token_id < 2:
            continue

        if next_slot >= TILES_PER_ROW * TILES_PER_ROW:
            print(f"[TextureExtractor] Warning: Atlas slot limit ({TILES_PER_ROW * TILES_PER_ROW}) reached!")
            break

        tile_x = next_slot % TILES_PER_ROW
        tile_y = next_slot // TILES_PER_ROW
        pixel_x = tile_x * TILE_SIZE
        pixel_y = tile_y * TILE_SIZE

        tex_jar_path = get_preferred_texture_name(block_name, available_textures)
        tile_img = None

        if tex_jar_path:
            try:
                with zip_file.open(tex_jar_path) as img_file:
                    raw_img = Image.open(img_file).convert("RGBA")
                    # If texture is animated (vertical strip), crop top 16x16 frame
                    if raw_img.height > TILE_SIZE:
                        tile_img = raw_img.crop((0, 0, TILE_SIZE, TILE_SIZE))
                    elif raw_img.size == (TILE_SIZE, TILE_SIZE):
                        tile_img = raw_img
                    else:
                        tile_img = raw_img.resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
                    matched_count += 1
            except Exception:
                tile_img = None

        if tile_img is None:
            # Generate solid color fallback tile
            fallback_color_count += 1
            import hashlib
            digest = hashlib.md5(block_name.encode("utf-8")).hexdigest()
            h = int(digest[:6], 16)
            r = max(50, min(230, (h >> 16) & 0xFF))
            g = max(50, min(230, (h >> 8) & 0xFF))
            b = max(50, min(230, b & 0xFF))
            tile_img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (r, g, b, 255))

        # Calculate average color
        np_pixels = list(tile_img.getdata())
        r_sum, g_sum, b_sum, count = 0, 0, 0, 0
        for px in np_pixels:
            if px[3] > 32:  # ignore mostly transparent pixels
                r_sum += px[0]
                g_sum += px[1]
                b_sum += px[2]
                count += 1
        if count > 0:
            avg_color = f"#{r_sum//count:02x}{g_sum//count:02x}{b_sum//count:02x}"
        else:
            avg_color = "#808080"

        # Paste into atlas
        atlas_img.paste(tile_img, (pixel_x, pixel_y))

        # UV coordinates for Three.js (U: 0..1, V: 0..1 from bottom-left)
        u0 = tile_x / TILES_PER_ROW
        v0 = 1.0 - ((tile_y + 1) / TILES_PER_ROW)
        u1 = (tile_x + 1) / TILES_PER_ROW
        v1 = 1.0 - (tile_y / TILES_PER_ROW)

        uv_mapping[str(token_id)] = {
            "id": token_id,
            "name": block_name,
            "tileX": tile_x,
            "tileY": tile_y,
            "u0": round(u0, 6),
            "v0": round(v0, 6),
            "u1": round(u1, 6),
            "v1": round(v1, 6),
            "avgColor": avg_color
        }

        next_slot += 1

    # 5. Save Atlas PNG and UV JSON
    atlas_img.save(atlas_png_path, "PNG")
    with open(uv_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "tileSize": TILE_SIZE,
            "tilesPerRow": TILES_PER_ROW,
            "atlasSize": ATLAS_SIZE,
            "totalTokensMapped": len(uv_mapping),
            "matchedTexturesFromJar": matched_count,
            "fallbackGenerated": fallback_color_count,
            "blocks": uv_mapping
        }, f, indent=2)

    print(f"[TextureExtractor] Successfully built Texture Atlas:")
    print(f"  -> Atlas PNG: {atlas_png_path} ({ATLAS_SIZE}x{ATLAS_SIZE} px)")
    print(f"  -> UV Mapping: {uv_json_path} ({len(uv_mapping)} block tokens mapped)")
    print(f"  -> Matched {matched_count} official 16x16 PNG textures from JAR.")

    return atlas_png_path, uv_json_path


def main():
    jar_path = find_client_jar()
    if not jar_path:
        print("Error: Could not locate Minecraft 1.21.1 client JAR.")
        sys.exit(1)

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    build_texture_atlas(jar_path, assets_dir)


if __name__ == "__main__":
    main()

