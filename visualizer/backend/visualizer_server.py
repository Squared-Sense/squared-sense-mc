"""
Real-time Visualizer Backend Server for Squared Sense MC.
- Reads live observation tensors from the Shared Memory (MMF) IPC bus.
- Serves the Three.js frontend web app at http://localhost:8765.
- Serves the 512x512 official Texture Atlas and UV mappings.
- Broadcasts live multi-LOD frames over WebSocket ws://localhost:8765/ws.
"""

import argparse
import asyncio
import http.server
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Add project root and bridge to import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
if BRIDGE_DIR not in sys.path:
    sys.path.append(BRIDGE_DIR)

import mc_ipc_layout as layout
from mc_ipc_reader import MinecraftIpcReader

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))


class VoxelFrameExtractor:
    def __init__(self, reader: MinecraftIpcReader):
        self.reader = reader
        self.uv_data = self._load_uv_data()
        self.head_data = self._load_head_data()

        # Precompute grid meshgrid
        self.gx_grid, self.gy_grid, self.gz_grid = np.meshgrid(
            np.arange(layout.GRID_SIZE),
            np.arange(layout.GRID_SIZE),
            np.arange(layout.GRID_SIZE),
            indexing="ij"
        )

    def _load_uv_data(self) -> Dict:
        uv_file = os.path.join(ASSETS_DIR, "atlas_uvs.json")
        if os.path.exists(uv_file):
            try:
                with open(uv_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[VisualizerServer] Loaded UV atlas metadata ({len(data.get('blocks', {}))} blocks).")
                    return data.get("blocks", {})
            except Exception as e:
                print(f"[VisualizerServer] Error loading {uv_file}: {e}")
        return {}

    def _load_head_data(self) -> Dict:
        head_file = os.path.join(ASSETS_DIR, "entity_heads.json")
        if os.path.exists(head_file):
            try:
                with open(head_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"[VisualizerServer] Loaded Entity Head metadata ({len(data.get('heads', {}))} heads).")
                    return data
            except Exception as e:
                print(f"[VisualizerServer] Error loading {head_file}: {e}")
        return {}

    def extract_lod_voxels(self, active_slot: int, lod: int, header: Dict) -> List[Dict]:
        stride = header["strides"][lod]
        cardinal = header["cardinal"]
        foot_index = header["footIndex"]

        blocks = self.reader.read_block_tensors(active_slot, lod=lod)
        entities = self.reader.read_entity_tensors(active_slot, lod=lod)

        x_local = (self.gx_grid - 8) * stride
        y_local = (self.gy_grid - foot_index) * stride
        z_local = (self.gz_grid - 2) * stride

        if cardinal == "SOUTH":
            dx_grid = -x_local
            dz_grid = z_local
        elif cardinal == "WEST":
            dx_grid = -z_local
            dz_grid = -x_local
        elif cardinal == "NORTH":
            dx_grid = x_local
            dz_grid = -z_local
        else:  # EAST
            dx_grid = z_local
            dz_grid = x_local
        dy_grid = y_local

        block_ids = blocks["block_id"]
        occupancies = blocks["occupancy"]
        passabilities = blocks["passability"]
        hazards = blocks["hazard"]
        hardnesses = blocks["hardness"]
        lights = blocks["light_level"]
        flow_x = blocks["fluid_flow_x"]
        flow_z = blocks["fluid_flow_z"]

        ent_counts = entities["entity_count"]
        ent_hostiles = entities["is_hostile"]
        ent_healths = entities["total_health"]

        mask = (block_ids != 0) | (ent_counts > 0)
        indices = np.argwhere(mask)

        voxels = []
        for idx in indices:
            gx, gy, gz = int(idx[0]), int(idx[1]), int(idx[2])
            bid = int(block_ids[gx, gy, gz])

            uv_info = self.uv_data.get(str(bid), {})
            bname = uv_info.get("name", f"block:{bid}")
            tile_x = uv_info.get("tileX", 0)
            tile_y = uv_info.get("tileY", 0)
            avg_color = uv_info.get("avgColor", "#808080")

            ec = int(ent_counts[gx, gy, gz])
            eh = float(ent_hostiles[gx, gy, gz])
            hp = round(float(ent_healths[gx, gy, gz]), 1)
            etype = None
            if ec > 0:
                etype = "zombie" if eh > 0.0 else "cow"

            voxels.append({
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "dx": int(dx_grid[gx, gy, gz]),
                "dy": int(dy_grid[gx, gy, gz]),
                "dz": int(dz_grid[gx, gy, gz]),
                "id": bid,
                "name": bname,
                "tileX": tile_x,
                "tileY": tile_y,
                "avgColor": avg_color,
                "occ": round(float(occupancies[gx, gy, gz]), 2),
                "pass": round(float(passabilities[gx, gy, gz]), 2),
                "haz": round(float(hazards[gx, gy, gz]), 2),
                "hard": round(float(hardnesses[gx, gy, gz]), 1),
                "light": round(float(lights[gx, gy, gz]), 2),
                "flow": [round(float(flow_x[gx, gy, gz]), 2), round(float(flow_z[gx, gy, gz]), 2)],
                "entCount": ec,
                "hostile": eh,
                "health": hp,
                "entType": etype
            })

        return voxels

    def build_frame(self, active_slot: int, lod_mode: str = "0") -> Optional[Dict]:
        header = self.reader.read_header()
        if not header:
            return None

        frame_header = {
            "sequenceId": header["sequenceId"],
            "gameTick": header["gameTick"],
            "timestampNanos": header["timestampNanos"],
            "playerPos": [round(p, 2) for p in header["playerPos"]],
            "cameraYaw": round(header["cameraYaw"], 1),
            "cameraPitch": round(header["cameraPitch"], 1),
            "snappedYaw": round(header["snappedYaw"], 1),
            "cardinal": header["cardinal"],
            "footIndex": header["footIndex"],
            "strides": header["strides"],
            "vocabSize": header["vocabSize"],
            "lodMode": str(lod_mode)
        }

        if lod_mode == "all":
            # Multi-LOD Overlap View (returns all 3 LODs)
            b0 = self.extract_lod_voxels(active_slot, 0, header)
            b1 = self.extract_lod_voxels(active_slot, 1, header)
            b2 = self.extract_lod_voxels(active_slot, 2, header)
            return {
                "header": frame_header,
                "mode": "all",
                "lod0": b0,
                "lod1": b1,
                "lod2": b2,
                "stats": {
                    "lod0Count": len(b0),
                    "lod1Count": len(b1),
                    "lod2Count": len(b2)
                }
            }
        else:
            lod = int(lod_mode) if str(lod_mode).isdigit() else 0
            lod = max(0, min(2, lod))
            voxels = self.extract_lod_voxels(active_slot, lod, header)
            return {
                "header": frame_header,
                "mode": "single",
                "lod": lod,
                "stride": header["strides"][lod],
                "voxels": voxels,
                "stats": {
                    "activeVoxels": len(voxels)
                }
            }


    def generate_demo_frame(self, lod_mode: str = "0", tick: int = 0) -> Dict:
        """Generates a synthetic Minecraft observation frame for verification/testing without launching the game."""
        import math
        sway = math.sin(tick * 0.1) * 20.0
        header = {
            "sequenceId": tick,
            "gameTick": 12000 + tick,
            "timestampNanos": int(time.time() * 1e9),
            "playerPos": [128.5, 64.0, -256.5],
            "cameraYaw": round(sway, 1),
            "cameraPitch": round(-5.0 + math.cos(tick * 0.08) * 5.0, 1),
            "snappedYaw": 0.0,
            "cardinal": "SOUTH",
            "footIndex": 6,
            "strides": [1, 2, 4],
            "vocabSize": 1014,
            "lodMode": str(lod_mode)
        }

        def build_demo_lod_voxels(lod: int) -> List[Dict]:
            stride = header["strides"][lod]
            voxels = []
            # Grid bounds for demo
            step = 1 if lod == 0 else (2 if lod == 1 else 4)
            for gx in range(0, layout.GRID_SIZE, step):
                for gz in range(0, layout.GRID_SIZE, step):
                    # Ground layers
                    for gy in range(0, 6, step):
                        bid = 848 if gy < 2 else (338 if gy < 5 else 408) # stone, dirt, grass
                        haz = 0.0
                        light = 0.0 if gy < 4 else 0.8
                        hard = 3.0 if bid == 848 else (1.5 if bid == 338 else 0.6)
                        passability = 0.0
                        occ = 1.0

                        # Water pool in corner
                        if gy == 5 and 2 <= gx <= 5 and 7 <= gz <= 10:
                            bid = 969 # water_still
                            passability = 0.8
                            haz = 0.0

                        # Lava pool in opposite corner
                        if gy == 5 and 11 <= gx <= 14 and 11 <= gz <= 14:
                            bid = 480 # lava_still
                            haz = 1.0
                            light = 1.0
                            passability = 0.2

                        uv_info = self.uv_data.get(str(bid), {})
                        voxels.append({
                            "gx": gx, "gy": gy, "gz": gz,
                            "dx": -(gx - 8) * stride,
                            "dy": (gy - 6) * stride,
                            "dz": (gz - 2) * stride,
                            "id": bid,
                            "name": uv_info.get("name", f"block:{bid}"),
                            "tileX": uv_info.get("tileX", 0),
                            "tileY": uv_info.get("tileY", 0),
                            "avgColor": uv_info.get("avgColor", "#808080"),
                            "occ": occ, "pass": passability, "haz": haz,
                            "hard": hard, "light": light,
                            "flow": [0.0, 0.0],
                            "entCount": 0, "hostile": 0.0, "health": 0.0
                        })

            # Tree (trunk + leaves)
            for ty in range(6, 10):
                tgx, tgz = 10, 5
                bid = 580 # oak_log
                uv_info = self.uv_data.get(str(bid), {})
                voxels.append({
                    "gx": tgx, "gy": ty, "gz": tgz,
                    "dx": -(tgx - 8) * stride, "dy": (ty - 6) * stride, "dz": (tgz - 2) * stride,
                    "id": bid, "name": uv_info.get("name", "minecraft:oak_log"),
                    "tileX": uv_info.get("tileX", 0), "tileY": uv_info.get("tileY", 0),
                    "avgColor": uv_info.get("avgColor", "#675231"),
                    "occ": 1.0, "pass": 0.0, "haz": 0.0, "hard": 2.0, "light": 0.8,
                    "flow": [0.0, 0.0], "entCount": 0, "hostile": 0.0, "health": 0.0
                })
            # Canopy
            for lx in range(8, 13):
                for lz in range(3, 8):
                    for ly in range(9, 12):
                        if lx == 10 and lz == 5 and ly < 11:
                            continue
                        bid = 579 # oak_leaves
                        uv_info = self.uv_data.get(str(bid), {})
                        voxels.append({
                            "gx": lx, "gy": ly, "gz": lz,
                            "dx": -(lx - 8) * stride, "dy": (ly - 6) * stride, "dz": (lz - 2) * stride,
                            "id": bid, "name": uv_info.get("name", "minecraft:oak_leaves"),
                            "tileX": uv_info.get("tileX", 0), "tileY": uv_info.get("tileY", 0),
                            "avgColor": uv_info.get("avgColor", "#32591e"),
                            "occ": 0.6, "pass": 0.5, "haz": 0.0, "hard": 0.2, "light": 0.9,
                            "flow": [0.0, 0.0], "entCount": 0, "hostile": 0.0, "health": 0.0
                        })

            # Demo Mobs: Zombie, Skeleton, Creeper, Cow, Pig
            demo_mobs = [
                (8, 6, 9, "zombie", 1.0, 20.0),
                (3, 6, 8, "skeleton", 1.0, 20.0),
                (12, 6, 7, "creeper", 1.0, 20.0),
                (5, 6, 4, "cow", 0.0, 10.0),
                (11, 6, 3, "pig", 0.0, 10.0)
            ]
            for mgx, mgy, mgz, mtype, mhostile, mhp in demo_mobs:
                voxels.append({
                    "gx": mgx, "gy": mgy, "gz": mgz,
                    "dx": -(mgx - 8) * stride, "dy": (mgy - 6) * stride, "dz": (mgz - 2) * stride,
                    "id": 0, "name": f"entity:{mtype}",
                    "tileX": 0, "tileY": 0, "avgColor": "#000000",
                    "occ": 0.0, "pass": 1.0, "haz": 0.0, "hard": 0.0, "light": 0.8,
                    "flow": [0.0, 0.0],
                    "entCount": 1, "hostile": mhostile, "health": mhp,
                    "entType": mtype
                })

            return voxels

        if lod_mode == "all":
            b0 = build_demo_lod_voxels(0)
            b1 = build_demo_lod_voxels(1)
            b2 = build_demo_lod_voxels(2)
            return {
                "header": header,
                "mode": "all",
                "lod0": b0, "lod1": b1, "lod2": b2,
                "stats": {"lod0Count": len(b0), "lod1Count": len(b1), "lod2Count": len(b2)}
            }
        else:
            lod = int(lod_mode) if str(lod_mode).isdigit() else 0
            lod = max(0, min(2, lod))
            voxels = build_demo_lod_voxels(lod)
            return {
                "header": header,
                "mode": "single",
                "lod": lod,
                "stride": header["strides"][lod],
                "voxels": voxels,
                "stats": {"activeVoxels": len(voxels)}
            }


class VisualizerServer:
    def __init__(self, port: int = 8765, ws_port: int = 8766, shm_path: Optional[str] = None, demo: bool = False):
        self.port = port
        self.ws_port = ws_port
        self.demo = demo
        self.reader = MinecraftIpcReader(shm_path)
        self.extractor = VoxelFrameExtractor(self.reader)
        self.active_lod_mode = "0"  # "0", "1", "2", or "all"
        self.latest_frame: Optional[Dict] = None
        self.latest_seq = -1
        self.connected_ws: Set = set()
        self.running = False
        self.demo_tick = 0

    def start(self):
        self.running = True
        print(f"[VisualizerServer] Starting visualizer backend...")

        # 1. Start HTTP Static File + API server
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # 2. Start MMF Poller thread
        poll_thread = threading.Thread(target=self._run_mmf_poll, daemon=True)
        poll_thread.start()

        # 3. Start WebSocket server on main thread
        if HAS_WEBSOCKETS:
            asyncio.run(self._run_websocket_server())
        else:
            print("[VisualizerServer] Running in HTTP-only mode.")
            while self.running:
                time.sleep(1.0)

    def _run_mmf_poll(self):
        while self.running:
            if self.demo:
                self.demo_tick += 1
                frame = self.extractor.generate_demo_frame(self.active_lod_mode, self.demo_tick)
                self.latest_frame = frame
                self.latest_seq = self.demo_tick
                if HAS_WEBSOCKETS and self.connected_ws and hasattr(self, "async_loop"):
                    msg = json.dumps(frame)
                    for ws in list(self.connected_ws):
                        try:
                            asyncio.run_coroutine_threadsafe(ws.send(msg), self.async_loop)
                        except Exception:
                            pass
                time.sleep(0.1)
                continue

            if not self.reader.is_open():
                if not self.reader.open():
                    time.sleep(1.0)
                    continue
                print(f"[VisualizerServer] Connected to MMF IPC at {self.reader.shm_path}")

            header = self.reader.read_header()
            if header:
                seq = header["sequenceId"]
                if seq != self.latest_seq:
                    self.latest_seq = seq
                    slot = header["activeSlot"]
                    frame = self.extractor.build_frame(slot, self.active_lod_mode)
                    if frame:
                        self.latest_frame = frame
                        if HAS_WEBSOCKETS and self.connected_ws and hasattr(self, "async_loop"):
                            msg = json.dumps(frame)
                            for ws in list(self.connected_ws):
                                try:
                                    asyncio.run_coroutine_threadsafe(ws.send(msg), self.async_loop)
                                except Exception:
                                    pass
            time.sleep(0.02)  # 50 Hz poll

    async def _run_websocket_server(self):
        self.async_loop = asyncio.get_running_loop()
        async with websockets.serve(self._handle_ws, "0.0.0.0", self.ws_port):
            print(f"[VisualizerServer] WebSocket live stream active on ws://localhost:{self.ws_port}")
            while self.running:
                await asyncio.sleep(1.0)

    async def _handle_ws(self, websocket):
        self.connected_ws.add(websocket)
        print(f"[VisualizerServer] Visualizer frontend connected! (Clients: {len(self.connected_ws)})")
        try:
            if self.latest_frame:
                await websocket.send(json.dumps(self.latest_frame))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    cmd = data.get("cmd")
                    if cmd == "set_lod":
                        mode = str(data.get("lod", "0"))
                        self.active_lod_mode = mode
                        print(f"[VisualizerServer] Switched LOD mode to: {mode}")
                        if self.demo:
                            frame = self.extractor.generate_demo_frame(self.active_lod_mode, self.demo_tick)
                            self.latest_frame = frame
                            await websocket.send(json.dumps(frame))
                        else:
                            header = self.reader.read_header()
                            if header:
                                frame = self.extractor.build_frame(header["activeSlot"], self.active_lod_mode)
                                if frame:
                                    self.latest_frame = frame
                                    await websocket.send(json.dumps(frame))
                except Exception as e:
                    print(f"[VisualizerServer] WS client message error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_ws.discard(websocket)
            print(f"[VisualizerServer] Frontend disconnected. (Clients: {len(self.connected_ws)})")

    def _run_http_server(self):
        server_self = self

        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress request spam

            def end_headers(self):
                # Aggressively disable browser caching for all visualizer frontend files and assets
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def do_GET(self):
                # Enable CORS
                path = self.path.split("?")[0]

                if path == "/assets/texture_atlas.png":
                    png_file = os.path.join(ASSETS_DIR, "texture_atlas.png")
                    if os.path.exists(png_file):
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        with open(png_file, "rb") as f:
                            self.wfile.write(f.read())
                        return

                if path == "/assets/atlas_uvs.json":
                    json_file = os.path.join(ASSETS_DIR, "atlas_uvs.json")
                    if os.path.exists(json_file):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        with open(json_file, "rb") as f:
                            self.wfile.write(f.read())
                        return

                if path == "/assets/entity_heads.png":
                    png_file = os.path.join(ASSETS_DIR, "entity_heads.png")
                    if os.path.exists(png_file):
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        with open(png_file, "rb") as f:
                            self.wfile.write(f.read())
                        return

                if path == "/assets/entity_heads.json":
                    json_file = os.path.join(ASSETS_DIR, "entity_heads.json")
                    if os.path.exists(json_file):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        with open(json_file, "rb") as f:
                            self.wfile.write(f.read())
                        return

                if path == "/api/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    status = {
                        "connected": server_self.reader.is_open(),
                        "shmPath": server_self.reader.shm_path,
                        "latestSeq": server_self.latest_seq,
                        "activeLodMode": server_self.active_lod_mode,
                        "connectedClients": len(server_self.connected_ws)
                    }
                    self.wfile.write(json.dumps(status).encode("utf-8"))
                    return

                if path == "/api/frame":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(server_self.latest_frame or {}).encode("utf-8"))
                    return

                # Default: serve frontend files
                super().do_GET()

        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), CustomHandler)
        print(f"[VisualizerServer] Frontend Web App available at: http://localhost:{self.port}")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Squared Sense MC 3D Visualizer Server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP frontend port (default: 8765)")
    parser.add_argument("--ws-port", type=int, default=8766, help="WebSocket live port (default: 8766)")
    parser.add_argument("--shm-path", type=str, default=None, help="Custom shared memory file path")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo frames when Minecraft is not active")
    args = parser.parse_args()

    server = VisualizerServer(port=args.port, ws_port=args.ws_port, shm_path=args.shm_path, demo=args.demo)
    server.start()


if __name__ == "__main__":
    main()
