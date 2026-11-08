/**
 * Squared Sense MC — 3D Observation Visualizer Application
 *
 * Real-time 3D voxel observation visualizer using Three.js Hardware Instancing:
 * - Option B: Minecraft Official Textures mapped via 512x512 Texture Atlas.
 * - Mob Preview Heads: 3D mob heads with official face textures (Zombie, Skeleton, Creeper, Cow, Pig, etc.).
 * - Multi-LOD support: B0 (1m solid), B1 (2m semi-transparent), B2 (4m wireframe).
 * - Physical channel heatmaps: Hazard, Passability, Light Level, Hardness.
 * - Entity overlays (E0, E1, E2): green for passive, pulsing red for hostiles.
 * - Player head sightline ray and foot anchor box.
 * - Interactive X/Y/Z cross-section clipping sliders.
 * - Real-time WebSocket streaming (ws://localhost:8766) with HTTP polling fallback.
 */

(function () {
    'use strict';

    // --- Configuration & Constants ---
    const ATLAS_TILES_PER_ROW = 32;
    const HEAD_TILES_PER_ROW = 8;
    const MAX_INSTANCES = 4096;
    const GRID_SIZE = 16;

    // --- State Variables ---
    let currentLodMode = '0'; // '0', '1', '2', or 'all'
    let currentRenderMode = 'texture'; // 'texture', 'hazard', 'passability', 'light', 'hardness'
    let showEntities = true;
    let showEntityHeads = true;
    let showSightline = true;
    let showPlayerBox = true;
    let showGrid = true;

    // Slice clipping range (0..15 in grid coordinates)
    const slices = {
        minX: 0, maxX: 15,
        minY: 0, maxY: 15,
        minZ: 0, maxZ: 15
    };

    let latestFrameData = null;
    let activeVoxelList = []; // For mouse raycasting / tooltip inspection
    let activeEntityList = []; // For entity heads tracking & orientation
    let uvAtlasMetadata = {};
    let entityHeadMetadata = {};
    let entityHeadAliases = {};

    let ws = null;
    let wsReconnectTimer = null;
    let httpPollTimer = null;

    // Three.js Core Objects
    let scene, camera, renderer, controls;
    let gridHelper;
    let playerGroup, playerArrow;
    let sightlineGroup, sightlineMesh, sightlineTip;
    let atlasTexture = null;
    let entityHeadTexture = null;

    // Instanced Meshes
    let meshLOD0 = null;
    let meshLOD1 = null;
    let meshLOD2 = null;
    let meshLOD2Wire = null;
    let meshEntities = null;
    let meshEntityHeads = null;

    // Custom Shader Uniforms
    const shaderUniforms = {
        uUseAtlas: { value: 1.0 }
    };

    // Reusable Math / Temp Objects
    const dummyObj = new THREE.Object3D();
    const tempMatrix = new THREE.Matrix4();
    const tempColor = new THREE.Color();
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(-1000, -1000);

    // --- Initialization Entry Point ---
    async function init() {
        initThreeScene();
        initPlayerAndHelpers();
        initInstancedMeshes();
        initUIEventListeners();

        // 1. Load UV Metadata & Textures
        await Promise.all([
            loadAtlasMetadata(),
            loadEntityHeadMetadata()
        ]);
        loadAtlasTexture();
        loadEntityHeadTexture();

        // 2. Connect to Live Stream
        connectWebSocket();

        // 3. Start Animation Loop
        requestAnimationFrame(renderLoop);
    }

    // --- 1. Three.js Scene Setup ---
    function initThreeScene() {
        const container = document.getElementById('canvas-container');

        // Scene & Fog
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0d0f14);
        scene.fog = new THREE.FogExp2(0x0d0f14, 0.01);

        // Perspective Camera
        camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 500);
        camera.position.set(16, 20, 24);

        // WebGL Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.LinearToneMapping;
        container.appendChild(renderer.domElement);

        // OrbitControls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 3, 0);
        controls.maxDistance = 180;
        controls.minDistance = 1.0;

        // Lighting (Directional + Ambient + Fill)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.75);
        dirLight1.position.set(30, 50, 25);
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0x7799cc, 0.35);
        dirLight2.position.set(-25, 20, -25);
        scene.add(dirLight2);

        // Ground Reference Grid
        gridHelper = new THREE.GridHelper(32, 32, 0x00d4ff, 0x1f2937);
        gridHelper.position.y = -0.5;
        scene.add(gridHelper);

        // Window Resize Listener
        window.addEventListener('resize', onWindowResize);
    }

    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // --- 2. Player Hitbox & Sightline Laser Helpers ---
    function initPlayerAndHelpers() {
        // Player Foot Anchor & Hitbox (0.6m x 1.8m x 0.6m)
        playerGroup = new THREE.Group();

        // Translucent Body Box
        const bodyGeo = new THREE.BoxGeometry(0.6, 1.8, 0.6);
        const bodyMat = new THREE.MeshBasicMaterial({
            color: 0x00d4ff,
            transparent: true,
            opacity: 0.25,
            depthWrite: false
        });
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = 0.9;
        playerGroup.add(bodyMesh);

        // Wireframe Outline
        const edgesGeo = new THREE.EdgesGeometry(bodyGeo);
        const edgesMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 });
        const bodyOutline = new THREE.LineSegments(edgesGeo, edgesMat);
        bodyOutline.position.y = 0.9;
        playerGroup.add(bodyOutline);

        // Facing Arrow on Player Head
        const arrowDir = new THREE.Vector3(0, 0, 1);
        playerArrow = new THREE.ArrowHelper(arrowDir, new THREE.Vector3(0, 1.62, 0), 1.2, 0x00d4ff, 0.3, 0.15);
        playerGroup.add(playerArrow);

        scene.add(playerGroup);

        // Sightline Laser Ray (From Player Eyes 1.62m to Distance)
        sightlineGroup = new THREE.Group();

        const laserGeo = new THREE.BufferGeometry();
        const laserPositions = new Float32Array([0, 1.62, 0, 0, 1.62, 20]);
        laserGeo.setAttribute('position', new THREE.BufferAttribute(laserPositions, 3));
        const laserMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 });
        sightlineMesh = new THREE.Line(laserGeo, laserMat);
        sightlineGroup.add(sightlineMesh);

        // Glowing Laser Target Tip
        const tipGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const tipMat = new THREE.MeshBasicMaterial({ color: 0x00d4ff });
        sightlineTip = new THREE.Mesh(tipGeo, tipMat);
        sightlineTip.position.set(0, 1.62, 20);
        sightlineGroup.add(sightlineTip);

        scene.add(sightlineGroup);
    }

    // --- 3. Instanced Meshes & Custom Texture Atlas Shader ---
    function createPlaceholderTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 2;
        canvas.height = 2;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, 2, 2);
        const tex = new THREE.CanvasTexture(canvas);
        tex.magFilter = THREE.NearestFilter;
        tex.minFilter = THREE.NearestFilter;
        return tex;
    }

    function createInstancedAtlasMaterial(isTransparent = false, opacityVal = 1.0) {
        const initialTex = atlasTexture || createPlaceholderTexture();
        const material = new THREE.MeshLambertMaterial({
            map: initialTex,
            transparent: isTransparent,
            opacity: opacityVal,
            depthWrite: !isTransparent,
            alphaTest: isTransparent ? 0.0 : 0.05
        });

        material.onBeforeCompile = (shader) => {
            // Add custom instanced UV offset attribute
            shader.vertexShader = `
                attribute vec2 instanceUVOffset;
                varying vec2 vAtlasUV;
            ` + shader.vertexShader;

            shader.vertexShader = shader.vertexShader.replace(
                '#include <uv_vertex>',
                `#include <uv_vertex>
                 // Clamp face UV slightly inward to avoid sub-pixel bleeding
                 vec2 clampedUv = clamp(uv, 0.005, 0.995);
                 vAtlasUV = instanceUVOffset + clampedUv * (1.0 / 32.0);`
            );

            // Fragment Shader: Sample atlas when uUseAtlas > 0.5, else use instanceColor
            shader.fragmentShader = `
                varying vec2 vAtlasUV;
                uniform float uUseAtlas;
            ` + shader.fragmentShader;

            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <map_fragment>',
                `#ifdef USE_MAP
                    if (uUseAtlas > 0.5) {
                        vec4 sampledColor = texture2D(map, vAtlasUV);
                        diffuseColor *= sampledColor;
                    }
                #endif`
            );

            // Connect uniform
            shader.uniforms.uUseAtlas = shaderUniforms.uUseAtlas;
            material.userData.shader = shader;
        };

        material.customProgramCacheKey = () => `instanced_atlas_mat_${isTransparent}_${opacityVal}`;
        return material;
    }

    function createEntityHeadMaterial() {
        const initialTex = entityHeadTexture || createPlaceholderTexture();
        const material = new THREE.MeshLambertMaterial({
            map: initialTex,
            transparent: true,
            alphaTest: 0.1
        });

        material.onBeforeCompile = (shader) => {
            shader.vertexShader = `
                attribute vec2 instanceHeadUV;
                varying vec2 vHeadUV;
                varying vec3 vHeadNormal;
            ` + shader.vertexShader;

            shader.vertexShader = shader.vertexShader.replace(
                '#include <uv_vertex>',
                `#include <uv_vertex>
                 vec2 clampedUv = clamp(uv, 0.005, 0.995);
                 vHeadUV = instanceHeadUV + clampedUv * (1.0 / 8.0);
                 vHeadNormal = normal;`
            );

            shader.fragmentShader = `
                varying vec2 vHeadUV;
                varying vec3 vHeadNormal;
            ` + shader.fragmentShader;

            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <map_fragment>',
                `#ifdef USE_MAP
                    vec4 headTex = texture2D(map, vHeadUV);
                    // Front face (+Z) has full brightness; sides, top, and back have 72% shade
                    if (vHeadNormal.z > 0.5) {
                        diffuseColor *= headTex;
                    } else {
                        diffuseColor *= headTex * vec4(0.72, 0.72, 0.72, 1.0);
                    }
                #endif`
            );

            material.userData.shader = shader;
        };

        material.customProgramCacheKey = () => 'instanced_entity_head_mat';
        return material;
    }

    function initInstancedMeshes() {
        // --- LOD 0 InstancedMesh (1x1x1 solid cubes) ---
        const geoLOD0 = new THREE.BoxGeometry(1.0, 1.0, 1.0);
        const uvOffsets0 = new Float32Array(MAX_INSTANCES * 2);
        const uvAttr0 = new THREE.InstancedBufferAttribute(uvOffsets0, 2);
        uvAttr0.setUsage(THREE.DynamicDrawUsage);
        geoLOD0.setAttribute('instanceUVOffset', uvAttr0);

        const matLOD0 = createInstancedAtlasMaterial(false, 1.0);
        meshLOD0 = new THREE.InstancedMesh(geoLOD0, matLOD0, MAX_INSTANCES);
        meshLOD0.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 3).fill(1.0), 3);
        meshLOD0.count = 0;
        scene.add(meshLOD0);

        // --- LOD 1 InstancedMesh (2x2x2 cubes) ---
        const geoLOD1 = new THREE.BoxGeometry(2.0, 2.0, 2.0);
        const uvOffsets1 = new Float32Array(MAX_INSTANCES * 2);
        const uvAttr1 = new THREE.InstancedBufferAttribute(uvOffsets1, 2);
        uvAttr1.setUsage(THREE.DynamicDrawUsage);
        geoLOD1.setAttribute('instanceUVOffset', uvAttr1);

        const matLOD1 = createInstancedAtlasMaterial(true, 0.35);
        meshLOD1 = new THREE.InstancedMesh(geoLOD1, matLOD1, MAX_INSTANCES);
        meshLOD1.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 3).fill(1.0), 3);
        meshLOD1.count = 0;
        meshLOD1.visible = false;
        scene.add(meshLOD1);

        // --- LOD 2 InstancedMesh Solid (4x4x4 cubes) ---
        const geoLOD2 = new THREE.BoxGeometry(4.0, 4.0, 4.0);
        const uvOffsets2 = new Float32Array(MAX_INSTANCES * 2);
        const uvAttr2 = new THREE.InstancedBufferAttribute(uvOffsets2, 2);
        uvAttr2.setUsage(THREE.DynamicDrawUsage);
        geoLOD2.setAttribute('instanceUVOffset', uvAttr2);

        const matLOD2 = createInstancedAtlasMaterial(false, 1.0);
        meshLOD2 = new THREE.InstancedMesh(geoLOD2, matLOD2, MAX_INSTANCES);
        meshLOD2.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 3).fill(1.0), 3);
        meshLOD2.count = 0;
        meshLOD2.visible = false;
        scene.add(meshLOD2);

        // --- LOD 2 InstancedMesh Wireframe Bounds (For Multi-LOD Overlap View) ---
        const geoLOD2Wire = new THREE.BoxGeometry(4.0, 4.0, 4.0);
        const matLOD2Wire = new THREE.MeshBasicMaterial({
            wireframe: true,
            color: 0x00d4ff,
            transparent: true,
            opacity: 0.45
        });
        meshLOD2Wire = new THREE.InstancedMesh(geoLOD2Wire, matLOD2Wire, MAX_INSTANCES);
        meshLOD2Wire.count = 0;
        meshLOD2Wire.visible = false;
        scene.add(meshLOD2Wire);

        // --- Entity Wireframe Boxes (Hostile = Pulsing Red, Passive = Bright Green) ---
        const geoEnt = new THREE.BoxGeometry(1.04, 1.04, 1.04);
        const matEnt = new THREE.MeshBasicMaterial({
            wireframe: true,
            transparent: true,
            opacity: 0.9
        });
        meshEntities = new THREE.InstancedMesh(geoEnt, matEnt, MAX_INSTANCES);
        meshEntities.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 3).fill(1.0), 3);
        meshEntities.count = 0;
        scene.add(meshEntities);

        // --- Entity 3D Mob Preview Heads (Front face = official mob face) ---
        const geoHead = new THREE.BoxGeometry(0.55, 0.55, 0.55);
        const headUvOffsets = new Float32Array(MAX_INSTANCES * 2);
        const headUvAttr = new THREE.InstancedBufferAttribute(headUvOffsets, 2);
        headUvAttr.setUsage(THREE.DynamicDrawUsage);
        geoHead.setAttribute('instanceHeadUV', headUvAttr);

        const matHead = createEntityHeadMaterial();
        meshEntityHeads = new THREE.InstancedMesh(geoHead, matHead, MAX_INSTANCES);
        meshEntityHeads.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_INSTANCES * 3).fill(1.0), 3);
        meshEntityHeads.count = 0;
        scene.add(meshEntityHeads);
    }

    // --- 4. Asset Loading (Atlas PNG + UV JSON + Entity Heads) ---
    async function loadAtlasMetadata() {
        try {
            const resp = await fetch('/assets/atlas_uvs.json');
            if (resp.ok) {
                const data = await resp.json();
                uvAtlasMetadata = data.blocks || {};
                console.log(`[Visualizer] Loaded metadata for ${Object.keys(uvAtlasMetadata).length} blocks.`);
            }
        } catch (e) {
            console.warn('[Visualizer] Could not load /assets/atlas_uvs.json:', e);
        }
    }

    async function loadEntityHeadMetadata() {
        try {
            const resp = await fetch('/assets/entity_heads.json');
            if (resp.ok) {
                const data = await resp.json();
                entityHeadMetadata = data.heads || {};
                entityHeadAliases = data.aliases || {};
                console.log(`[Visualizer] Loaded ${Object.keys(entityHeadMetadata).length} mob preview heads.`);
            }
        } catch (e) {
            console.warn('[Visualizer] Could not load /assets/entity_heads.json:', e);
        }
    }

    function loadAtlasTexture() {
        const loader = new THREE.TextureLoader();
        atlasTexture = loader.load(
            '/assets/texture_atlas.png',
            (tex) => {
                tex.magFilter = THREE.NearestFilter;
                tex.minFilter = THREE.NearestFilter;
                tex.generateMipmaps = false;
                tex.flipY = true;
                console.log('[Visualizer] Official Texture Atlas loaded successfully.');

                // Apply texture to materials
                meshLOD0.material.map = tex;
                meshLOD0.material.needsUpdate = true;
                meshLOD1.material.map = tex;
                meshLOD1.material.needsUpdate = true;
                meshLOD2.material.map = tex;
                meshLOD2.material.needsUpdate = true;
            },
            undefined,
            (err) => {
                console.error('[Visualizer] Failed to load /assets/texture_atlas.png:', err);
            }
        );
    }

    function loadEntityHeadTexture() {
        const loader = new THREE.TextureLoader();
        entityHeadTexture = loader.load(
            '/assets/entity_heads.png',
            (tex) => {
                tex.magFilter = THREE.NearestFilter;
                tex.minFilter = THREE.NearestFilter;
                tex.generateMipmaps = false;
                tex.flipY = true;
                console.log('[Visualizer] Official Entity Head Atlas loaded successfully.');

                if (meshEntityHeads && meshEntityHeads.material) {
                    meshEntityHeads.material.map = tex;
                    meshEntityHeads.material.needsUpdate = true;
                }
            },
            undefined,
            (err) => {
                console.error('[Visualizer] Failed to load /assets/entity_heads.png:', err);
            }
        );
    }

    // --- 5. Frame Update & Voxel Upload Logic ---
    function handleNewFrame(frame) {
        if (!frame || !frame.header) return;
        latestFrameData = frame;

        const header = frame.header;

        // 1. Update Telemetry HUD
        updateTelemetryHUD(header, frame);

        // 2. Update Player Facing & Sightline
        updatePlayerSightline(header);

        // 3. Render Voxels according to mode
        renderFrameVoxels(frame);
    }

    function updateTelemetryHUD(header, frame) {
        document.getElementById('stat-tick').textContent = header.gameTick ?? 0;
        if (header.playerPos) {
            document.getElementById('stat-pos').textContent =
                `${header.playerPos[0].toFixed(1)}, ${header.playerPos[1].toFixed(1)}, ${header.playerPos[2].toFixed(1)}`;
        }
        document.getElementById('stat-cardinal').textContent = header.cardinal ?? 'SOUTH';
        document.getElementById('stat-yaw-pitch').textContent =
            `${(header.cameraYaw ?? 0).toFixed(1)}° / ${(header.cameraPitch ?? 0).toFixed(1)}°`;
        document.getElementById('stat-foot-idx').textContent = header.footIndex ?? 6;

        let totalVoxels = 0;
        let totalEntities = 0;

        if (frame.mode === 'all') {
            totalVoxels = (frame.lod0?.length || 0) + (frame.lod1?.length || 0) + (frame.lod2?.length || 0);
            (frame.lod0 || []).forEach(v => { if (v.entCount > 0) totalEntities += v.entCount; });
        } else {
            totalVoxels = frame.voxels?.length || 0;
            (frame.voxels || []).forEach(v => { if (v.entCount > 0) totalEntities += v.entCount; });
        }

        document.getElementById('stat-voxels').textContent = totalVoxels;
        document.getElementById('stat-entities').textContent = totalEntities;
    }

    function updatePlayerSightline(header) {
        const yaw = header.cameraYaw ?? 0;
        const pitch = header.cameraPitch ?? 0;

        // Minecraft Angle Conversion:
        // yaw 0 = South (+Z), 90 = West (-X), 180 = North (-Z), 270 = East (+X)
        const radYaw = yaw * (Math.PI / 180.0);
        const radPitch = pitch * (Math.PI / 180.0);

        const vx = -Math.sin(radYaw) * Math.cos(radPitch);
        const vy = -Math.sin(radPitch);
        const vz = Math.cos(radYaw) * Math.cos(radPitch);

        // Update Player Arrow Orientation (Horizontal Only)
        const horizLen = Math.hypot(vx, vz);
        if (horizLen > 0.001) {
            playerArrow.setDirection(new THREE.Vector3(vx / horizLen, 0, vz / horizLen));
        }

        // Update Sightline Laser Ray (24 meters reach)
        const rayLen = 24.0;
        const positions = sightlineMesh.geometry.attributes.position.array;
        positions[0] = 0;
        positions[1] = 1.62;
        positions[2] = 0;
        positions[3] = vx * rayLen;
        positions[4] = 1.62 + vy * rayLen;
        positions[5] = vz * rayLen;
        sightlineMesh.geometry.attributes.position.needsUpdate = true;

        sightlineTip.position.set(vx * rayLen, 1.62 + vy * rayLen, vz * rayLen);
    }

    // --- 6. Voxel Upload & Shader Channel Heatmaps ---
    function isVoxelInSlice(vox) {
        return vox.gx >= slices.minX && vox.gx <= slices.maxX &&
               vox.gy >= slices.minY && vox.gy <= slices.maxY &&
               vox.gz >= slices.minZ && vox.gz <= slices.maxZ;
    }

    function getVoxelColor(vox, mode) {
        if (mode === 'texture') {
            return tempColor.setRGB(1.0, 1.0, 1.0);
        }

        if (mode === 'hazard') {
            const haz = vox.haz || 0.0;
            if (haz <= 0.001) {
                return tempColor.setRGB(0.18, 0.22, 0.28); // Cool dark slate
            } else if (haz < 0.5) {
                // Warning Yellow-Orange
                const t = haz / 0.5;
                return tempColor.setRGB(1.0, 0.85 - t * 0.35, 0.0);
            } else {
                // Extreme Danger Glowing Red
                const t = (haz - 0.5) / 0.5;
                return tempColor.setRGB(1.0, 0.5 - t * 0.45, 0.0);
            }
        }

        if (mode === 'passability') {
            const pass = vox.pass !== undefined ? vox.pass : 0.0;
            if (pass >= 0.99) {
                // Passable / Open Air: Emerald Green
                return tempColor.setRGB(0.15, 0.92, 0.3);
            } else if (pass > 0.05) {
                // Partial / Climbable: Amber Yellow
                return tempColor.setRGB(0.95, 0.75, 0.1);
            } else {
                // Solid Obstacle: Dark Red
                return tempColor.setRGB(0.75, 0.12, 0.15);
            }
        }

        if (mode === 'light') {
            const light = Math.max(0.0, Math.min(1.0, vox.light || 0.0));
            // 0 = Midnight Black/Navy, 1 = Warm Bright Sunlight
            return tempColor.setRGB(
                0.04 + light * 0.96,
                0.04 + light * 0.92,
                0.10 + light * 0.55
            );
        }

        if (mode === 'hardness') {
            const hard = vox.hard !== undefined ? vox.hard : 0.0;
            if (hard < 0) {
                // Unbreakable Bedrock: Deep Obsidian Purple
                return tempColor.setRGB(0.35, 0.05, 0.45);
            } else if (hard === 0) {
                // Instant Break: Cyan Tint
                return tempColor.setRGB(0.2, 0.8, 0.9);
            } else if (hard <= 2.0) {
                // Soft (Dirt / Wood): Earthy Greenish Gold
                return tempColor.setRGB(0.4, 0.75, 0.2);
            } else if (hard <= 5.0) {
                // Stone / Iron: Slate Blue-Gray
                return tempColor.setRGB(0.5, 0.55, 0.65);
            } else {
                // Very Hard (Obsidian): Dark Crimson
                return tempColor.setRGB(0.75, 0.15, 0.35);
            }
        }

        return tempColor.setRGB(1.0, 1.0, 1.0);
    }

    function populateInstancedMesh(mesh, voxels, stride, scaleMultiplier = 1.0) {
        if (!mesh) return 0;

        const uvAttr = mesh.geometry.getAttribute('instanceUVOffset');
        const uvArray = uvAttr ? uvAttr.array : null;
        let count = 0;

        for (let i = 0; i < voxels.length && count < MAX_INSTANCES; i++) {
            const vox = voxels[i];
            if (!isVoxelInSlice(vox)) continue;

            // Set Position & Scale
            dummyObj.position.set(vox.dx, vox.dy, vox.dz);
            dummyObj.scale.set(stride * scaleMultiplier, stride * scaleMultiplier, stride * scaleMultiplier);
            dummyObj.updateMatrix();
            mesh.setMatrixAt(count, dummyObj.matrix);

            // Set UV Tile Offset (Atlas Tile: tileX / 32, 1.0 - (tileY + 1) / 32)
            if (uvArray) {
                const u0 = (vox.tileX || 0) / ATLAS_TILES_PER_ROW;
                const v0 = 1.0 - ((vox.tileY || 0) + 1.0) / ATLAS_TILES_PER_ROW;
                uvArray[count * 2 + 0] = u0;
                uvArray[count * 2 + 1] = v0;
            }

            // Set Instance Color
            const col = getVoxelColor(vox, currentRenderMode);
            mesh.setColorAt(count, col);

            count++;
        }

        mesh.count = count;
        mesh.instanceMatrix.needsUpdate = true;
        if (uvAttr) uvAttr.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

        return count;
    }

    function renderFrameVoxels(frame) {
        // Toggle shader atlas usage based on render mode
        shaderUniforms.uUseAtlas.value = (currentRenderMode === 'texture') ? 1.0 : 0.0;

        activeVoxelList = [];
        const entityVoxels = [];

        if (frame.mode === 'all') {
            // Multi-LOD Overlap View Mode:
            // - B0: Solid 1x1x1 cubes
            // - B1: Semi-transparent 2x2x2 cubes (35% opacity)
            // - B2: Wireframe bounds 4x4x4 cubes

            // 1. LOD 0 (Solid 1m)
            meshLOD0.visible = true;
            const b0Voxels = (frame.lod0 || []).filter(isVoxelInSlice);
            populateInstancedMesh(meshLOD0, b0Voxels, 1.0, 1.0);
            activeVoxelList = b0Voxels;

            // 2. LOD 1 (Semi-transparent 2m)
            meshLOD1.visible = true;
            meshLOD1.material.opacity = 0.35;
            meshLOD1.material.transparent = true;
            meshLOD1.material.depthWrite = false;
            const b1Voxels = (frame.lod1 || []).filter(isVoxelInSlice);
            populateInstancedMesh(meshLOD1, b1Voxels, 2.0, 1.0);

            // 3. LOD 2 (Wireframe Bounds 4m)
            meshLOD2.visible = false;
            meshLOD2Wire.visible = true;
            const b2Voxels = (frame.lod2 || []).filter(isVoxelInSlice);
            populateInstancedMesh(meshLOD2Wire, b2Voxels, 4.0, 1.0);

            // Collect entities across LOD 0
            (frame.lod0 || []).forEach(v => {
                if (v.entCount > 0 && isVoxelInSlice(v)) entityVoxels.push({ vox: v, stride: 1.0 });
            });

        } else {
            // Single LOD View Mode (0, 1, or 2)
            const lod = frame.lod || 0;
            const stride = frame.stride || 1;
            const voxels = (frame.voxels || []).filter(isVoxelInSlice);
            activeVoxelList = voxels;

            // Hide inactive meshes
            meshLOD0.visible = (lod === 0);
            meshLOD1.visible = (lod === 1);
            meshLOD2.visible = (lod === 2);
            meshLOD2Wire.visible = false;

            if (lod === 0) {
                populateInstancedMesh(meshLOD0, voxels, 1.0, 1.0);
            } else if (lod === 1) {
                // In single LOD 1 view, render solid
                meshLOD1.material.opacity = 1.0;
                meshLOD1.material.transparent = false;
                meshLOD1.material.depthWrite = true;
                populateInstancedMesh(meshLOD1, voxels, 2.0, 1.0);
            } else if (lod === 2) {
                // In single LOD 2 view, render solid
                meshLOD2.material.opacity = 1.0;
                meshLOD2.material.transparent = false;
                meshLOD2.material.depthWrite = true;
                populateInstancedMesh(meshLOD2, voxels, 4.0, 1.0);
            }

            voxels.forEach(v => {
                if (v.entCount > 0) entityVoxels.push({ vox: v, stride: stride });
            });
        }

        activeEntityList = entityVoxels;

        // 4. Update Entity Overlays (Wireframes & 3D Mob Preview Heads)
        updateEntityBoxes(entityVoxels);
        updateEntityHeads(entityVoxels);
    }

    function updateEntityBoxes(entityList) {
        if (!showEntities || !meshEntities) {
            if (meshEntities) meshEntities.count = 0;
            return;
        }

        let count = 0;
        for (let i = 0; i < entityList.length && count < MAX_INSTANCES; i++) {
            const item = entityList[i];
            const vox = item.vox;
            const stride = item.stride;

            dummyObj.position.set(vox.dx, vox.dy, vox.dz);
            dummyObj.scale.set(stride * 1.05, stride * 1.05, stride * 1.05);
            dummyObj.updateMatrix();
            meshEntities.setMatrixAt(count, dummyObj.matrix);

            // Hostile = Glowing Red (#ff3366), Passive = Bright Green (#39ff14)
            if (vox.hostile > 0) {
                meshEntities.setColorAt(count, tempColor.setRGB(1.0, 0.2, 0.4));
            } else {
                meshEntities.setColorAt(count, tempColor.setRGB(0.22, 1.0, 0.08));
            }

            count++;
        }

        meshEntities.count = count;
        meshEntities.instanceMatrix.needsUpdate = true;
        if (meshEntities.instanceColor) meshEntities.instanceColor.needsUpdate = true;
    }

    function updateEntityHeads(entityList) {
        if (!showEntityHeads || !meshEntityHeads) {
            if (meshEntityHeads) meshEntityHeads.count = 0;
            return;
        }

        const headUvAttr = meshEntityHeads.geometry.getAttribute('instanceHeadUV');
        const headUvArray = headUvAttr ? headUvAttr.array : null;

        let count = 0;
        for (let i = 0; i < entityList.length && count < MAX_INSTANCES; i++) {
            const item = entityList[i];
            const vox = item.vox;
            const stride = item.stride;

            // Head scale: 0.55m at stride 1
            const headSize = 0.55 * stride;

            dummyObj.position.set(vox.dx, vox.dy, vox.dz);
            dummyObj.scale.set(headSize, headSize, headSize);

            // Orient head towards camera horizontally
            dummyObj.lookAt(camera.position.x, vox.dy, camera.position.z);
            dummyObj.updateMatrix();
            meshEntityHeads.setMatrixAt(count, dummyObj.matrix);

            // Determine UV tile from entity type
            let mobKey = (vox.entType || (vox.hostile > 0 ? 'zombie' : 'cow')).toLowerCase();
            let mobId = entityHeadAliases[mobKey] || entityHeadAliases[vox.name] || (vox.hostile > 0 ? 1 : 10);
            let headInfo = entityHeadMetadata[String(mobId)] || entityHeadMetadata['1'];

            if (headUvArray && headInfo) {
                const u0 = headInfo.tileX / HEAD_TILES_PER_ROW;
                const v0 = 1.0 - (headInfo.tileY + 1.0) / HEAD_TILES_PER_ROW;
                headUvArray[count * 2 + 0] = u0;
                headUvArray[count * 2 + 1] = v0;
            }

            meshEntityHeads.setColorAt(count, tempColor.setRGB(1.0, 1.0, 1.0));
            count++;
        }

        meshEntityHeads.count = count;
        meshEntityHeads.instanceMatrix.needsUpdate = true;
        if (headUvAttr) headUvAttr.needsUpdate = true;
        if (meshEntityHeads.instanceColor) meshEntityHeads.instanceColor.needsUpdate = true;
    }

    function orientEntityHeadsToCamera() {
        if (!showEntityHeads || !meshEntityHeads || meshEntityHeads.count === 0 || activeEntityList.length === 0) return;
        const total = Math.min(meshEntityHeads.count, activeEntityList.length);
        for (let i = 0; i < total; i++) {
            const item = activeEntityList[i];
            const vox = item.vox;
            const stride = item.stride;
            const headSize = 0.55 * stride;

            dummyObj.position.set(vox.dx, vox.dy, vox.dz);
            dummyObj.scale.set(headSize, headSize, headSize);
            dummyObj.lookAt(camera.position.x, vox.dy, camera.position.z);
            dummyObj.updateMatrix();
            meshEntityHeads.setMatrixAt(i, dummyObj.matrix);
        }
        meshEntityHeads.instanceMatrix.needsUpdate = true;
    }

    // --- 7. Interactive UI & Event Listeners ---
    function initUIEventListeners() {
        // LOD View Mode Buttons
        const lodButtons = document.querySelectorAll('[data-lod]');
        const lodLabel = document.getElementById('current-lod-label');

        lodButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                lodButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const chosenLod = btn.getAttribute('data-lod');
                currentLodMode = chosenLod;
                lodLabel.textContent = chosenLod === 'all' ? 'Multi-LOD Overlap' : `LOD ${chosenLod}`;

                // Send LOD change request to backend over WebSocket
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ cmd: 'set_lod', lod: chosenLod }));
                }

                // If offline or polling, re-render immediately
                if (latestFrameData) {
                    renderFrameVoxels(latestFrameData);
                }
            });
        });

        // Render Mode Heatmap Buttons
        const modeButtons = document.querySelectorAll('[data-mode]');
        modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                currentRenderMode = btn.getAttribute('data-mode');
                if (latestFrameData) {
                    renderFrameVoxels(latestFrameData);
                }
            });
        });

        // Overlays & Visual Helpers Toggles
        document.getElementById('toggle-entities').addEventListener('change', (e) => {
            showEntities = e.target.checked;
            meshEntities.visible = showEntities;
        });

        const toggleEntityHeads = document.getElementById('toggle-entity-heads');
        if (toggleEntityHeads) {
            toggleEntityHeads.addEventListener('change', (e) => {
                showEntityHeads = e.target.checked;
                meshEntityHeads.visible = showEntityHeads;
            });
        }

        document.getElementById('toggle-sightline').addEventListener('change', (e) => {
            showSightline = e.target.checked;
            sightlineGroup.visible = showSightline;
        });

        document.getElementById('toggle-player-box').addEventListener('change', (e) => {
            showPlayerBox = e.target.checked;
            playerGroup.visible = showPlayerBox;
        });

        document.getElementById('toggle-grid').addEventListener('change', (e) => {
            showGrid = e.target.checked;
            gridHelper.visible = showGrid;
        });

        // Interactive Slice Clipping Sliders
        const sliceXMin = document.getElementById('slice-x-min');
        const sliceXMax = document.getElementById('slice-x-max');
        const sliceYMin = document.getElementById('slice-y-min');
        const sliceYMax = document.getElementById('slice-y-max');
        const sliceZMin = document.getElementById('slice-z-min');
        const sliceZMax = document.getElementById('slice-z-max');

        const valSliceX = document.getElementById('val-slice-x');
        const valSliceY = document.getElementById('val-slice-y');
        const valSliceZ = document.getElementById('val-slice-z');

        function updateSliceValues() {
            let minX = parseInt(sliceXMin.value, 10);
            let maxX = parseInt(sliceXMax.value, 10);
            if (minX > maxX) { const t = minX; minX = maxX; maxX = t; }

            let minY = parseInt(sliceYMin.value, 10);
            let maxY = parseInt(sliceYMax.value, 10);
            if (minY > maxY) { const t = minY; minY = maxY; maxY = t; }

            let minZ = parseInt(sliceZMin.value, 10);
            let maxZ = parseInt(sliceZMax.value, 10);
            if (minZ > maxZ) { const t = minZ; minZ = maxZ; maxZ = t; }

            slices.minX = minX; slices.maxX = maxX;
            slices.minY = minY; slices.maxY = maxY;
            slices.minZ = minZ; slices.maxZ = maxZ;

            valSliceX.textContent = `${minX} - ${maxX}`;
            valSliceY.textContent = `${minY} - ${maxY}`;
            valSliceZ.textContent = `${minZ} - ${maxZ}`;

            if (latestFrameData) {
                renderFrameVoxels(latestFrameData);
            }
        }

        [sliceXMin, sliceXMax, sliceYMin, sliceYMax, sliceZMin, sliceZMax].forEach(slider => {
            slider.addEventListener('input', updateSliceValues);
        });

        document.getElementById('btn-reset-slices').addEventListener('click', () => {
            sliceXMin.value = 0; sliceXMax.value = 15;
            sliceYMin.value = 0; sliceYMax.value = 15;
            sliceZMin.value = 0; sliceZMax.value = 15;
            updateSliceValues();
        });

        // Camera View Shortcut Buttons
        document.getElementById('cam-reset').addEventListener('click', () => {
            camera.position.set(16, 20, 24);
            controls.target.set(0, 3, 0);
            controls.update();
        });

        document.getElementById('cam-top').addEventListener('click', () => {
            camera.position.set(0, 38, 0.01);
            controls.target.set(0, 0, 0);
            controls.update();
        });

        document.getElementById('cam-side').addEventListener('click', () => {
            camera.position.set(34, 4, 0);
            controls.target.set(0, 2, 0);
            controls.update();
        });

        document.getElementById('cam-align').addEventListener('click', () => {
            if (!latestFrameData || !latestFrameData.header) return;
            const yaw = latestFrameData.header.cameraYaw ?? 0;
            const pitch = latestFrameData.header.cameraPitch ?? 0;
            const radYaw = yaw * (Math.PI / 180.0);
            const radPitch = pitch * (Math.PI / 180.0);

            const vx = -Math.sin(radYaw) * Math.cos(radPitch);
            const vy = -Math.sin(radPitch);
            const vz = Math.cos(radYaw) * Math.cos(radPitch);

            // Third-person chase camera 5 meters behind player eyes
            const dist = 5.0;
            camera.position.set(-vx * dist, 1.62 - vy * dist + 0.6, -vz * dist);
            controls.target.set(vx * 10.0, 1.62 + vy * 10.0, vz * 10.0);
            controls.update();
        });

        // Mouse Hover Tooltip Raycasting
        window.addEventListener('pointermove', onPointerMove);
    }

    // --- 8. Mouse Raycasting & Tooltip Inspection ---
    function onPointerMove(event) {
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        const tooltip = document.getElementById('tooltip');
        if (!tooltip || activeVoxelList.length === 0) return;

        raycaster.setFromCamera(mouse, camera);

        // Determine currently active visible meshes
        const candidateMeshes = [];
        if (meshEntityHeads && meshEntityHeads.visible && meshEntityHeads.count > 0) {
            candidateMeshes.push(meshEntityHeads);
        }
        if (meshEntities && meshEntities.visible && meshEntities.count > 0) {
            candidateMeshes.push(meshEntities);
        }
        if (meshLOD0 && meshLOD0.visible) candidateMeshes.push(meshLOD0);
        if (meshLOD1 && meshLOD1.visible) candidateMeshes.push(meshLOD1);
        if (meshLOD2 && meshLOD2.visible) candidateMeshes.push(meshLOD2);

        const intersects = raycaster.intersectObjects(candidateMeshes);

        if (intersects.length > 0) {
            const hit = intersects[0];
            const instanceId = hit.instanceId;

            // If an entity head or entity wireframe was hit
            if ((hit.object === meshEntityHeads || hit.object === meshEntities) && instanceId !== undefined) {
                if (instanceId < activeEntityList.length) {
                    const item = activeEntityList[instanceId];
                    if (item && item.vox) {
                        showTooltip(event.clientX, event.clientY, item.vox);
                        return;
                    }
                }
            }

            // Otherwise, block terrain hit
            if (instanceId !== undefined && instanceId < activeVoxelList.length) {
                const vox = activeVoxelList[instanceId];
                if (vox) {
                    showTooltip(event.clientX, event.clientY, vox);
                    return;
                }
            }
        }

        hideTooltip();
    }

    function showTooltip(x, y, vox) {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;

        const bName = vox.name || `block:${vox.id}`;
        const occ = (vox.occ !== undefined) ? vox.occ.toFixed(2) : '1.0';
        const pass = (vox.pass !== undefined) ? vox.pass.toFixed(2) : '0.0';
        const haz = (vox.haz !== undefined) ? vox.haz.toFixed(2) : '0.0';
        const hard = (vox.hard !== undefined) ? vox.hard.toFixed(1) : '1.5';
        const light = (vox.light !== undefined) ? vox.light.toFixed(2) : '0.0';
        const lightLevel = Math.round((vox.light || 0.0) * 15);

        let entityHtml = '';
        if (vox.entCount > 0) {
            const isHostile = vox.hostile > 0;
            const mobKey = (vox.entType || (isHostile ? 'zombie' : 'cow')).toLowerCase();
            const mobId = entityHeadAliases[mobKey] || (isHostile ? 1 : 10);
            const headInfo = entityHeadMetadata[String(mobId)];
            const mobDisplayName = headInfo ? headInfo.name : (isHostile ? 'Hostile Mob' : 'Passive Animal');

            entityHtml = `
                <div class="tooltip-row ${isHostile ? 'hostile' : 'safe'}" style="margin-top: 6px; border-top: 1px dashed rgba(255,255,255,0.2); padding-top: 4px;">
                    <span class="label">Entity Preview:</span>
                    <span>${mobDisplayName} (${isHostile ? 'HOSTILE' : 'Passive'})</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Entity Health:</span>
                    <span>${vox.health || 20.0} HP</span>
                </div>
                <div class="tooltip-row">
                    <span class="label">Entity Count:</span>
                    <span>${vox.entCount}</span>
                </div>
            `;
        }

        tooltip.innerHTML = `
            <div class="tooltip-title">${bName} <span style="color: var(--text-muted); font-size: 10px;">(ID ${vox.id})</span></div>
            <div class="tooltip-row"><span class="label">Grid [gx,gy,gz]:</span> <span>${vox.gx}, ${vox.gy}, ${vox.gz}</span></div>
            <div class="tooltip-row"><span class="label">World [dx,dy,dz]:</span> <span>${vox.dx}m, ${vox.dy}m, ${vox.dz}m</span></div>
            <div class="tooltip-row"><span class="label">Occupancy:</span> <span>${occ}</span></div>
            <div class="tooltip-row ${vox.pass > 0.8 ? 'safe' : vox.pass < 0.2 ? 'danger' : ''}"><span class="label">Passability:</span> <span>${pass}</span></div>
            <div class="tooltip-row ${vox.haz > 0 ? 'danger' : 'safe'}"><span class="label">Hazard:</span> <span>${haz}</span></div>
            <div class="tooltip-row"><span class="label">Hardness:</span> <span>${hard}</span></div>
            <div class="tooltip-row"><span class="label">Light Level:</span> <span>${light} (${lightLevel}/15)</span></div>
            ${entityHtml}
        `;

        // Tooltip position clamped to screen bounds
        const tipWidth = 240;
        const tipHeight = 240;
        const posX = Math.min(x + 16, window.innerWidth - tipWidth - 20);
        const posY = Math.min(y + 16, window.innerHeight - tipHeight - 20);

        tooltip.style.left = `${posX}px`;
        tooltip.style.top = `${posY}px`;
        tooltip.classList.remove('hidden');
    }

    function hideTooltip() {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) tooltip.classList.add('hidden');
    }

    // --- 9. Connection Status & WebSocket Live Stream ---
    function updateConnectionStatus(isConnected) {
        const pill = document.getElementById('status-pill');
        const text = document.getElementById('status-text');
        if (!pill || !text) return;

        if (isConnected) {
            pill.className = 'status-pill connected';
            text.textContent = 'LIVE STREAM';
        } else {
            pill.className = 'status-pill disconnected';
            text.textContent = 'DISCONNECTED';
        }
    }

    function connectWebSocket() {
        const wsHost = window.location.hostname || 'localhost';
        const wsUrl = `ws://${wsHost}:8766`;
        console.log(`[Visualizer] Connecting to WebSocket: ${wsUrl}`);

        try {
            ws = new WebSocket(wsUrl);
        } catch (e) {
            console.warn('[Visualizer] WebSocket failed to initialize, starting HTTP polling fallback.', e);
            startHttpPollingFallback();
            return;
        }

        ws.onopen = () => {
            console.log('[Visualizer] Connected to live observation stream via WebSocket.');
            updateConnectionStatus(true);
            stopHttpPollingFallback();

            // Notify backend of current LOD selection
            ws.send(JSON.stringify({ cmd: 'set_lod', lod: currentLodMode }));
        };

        ws.onmessage = (event) => {
            try {
                const frame = JSON.parse(event.data);
                handleNewFrame(frame);
            } catch (e) {
                console.error('[Visualizer] Error parsing incoming frame JSON:', e);
            }
        };

        ws.onclose = () => {
            updateConnectionStatus(false);
            scheduleWsReconnect();
        };

        ws.onerror = (err) => {
            console.warn('[Visualizer] WebSocket error encountered, falling back to HTTP polling.');
            updateConnectionStatus(false);
            startHttpPollingFallback();
        };
    }

    function scheduleWsReconnect() {
        if (wsReconnectTimer) return;
        wsReconnectTimer = setTimeout(() => {
            wsReconnectTimer = null;
            connectWebSocket();
        }, 2000);
    }

    function startHttpPollingFallback() {
        if (httpPollTimer) return;
        console.log('[Visualizer] HTTP Polling fallback started (5 Hz).');
        httpPollTimer = setInterval(async () => {
            try {
                const resp = await fetch('/api/frame');
                if (resp.ok) {
                    const frame = await resp.json();
                    if (frame && frame.header) {
                        updateConnectionStatus(true);
                        handleNewFrame(frame);
                    }
                }
            } catch (e) {
                updateConnectionStatus(false);
            }
        }, 200);
    }

    function stopHttpPollingFallback() {
        if (httpPollTimer) {
            clearInterval(httpPollTimer);
            httpPollTimer = null;
        }
    }

    // --- 10. Animation & Render Loop ---
    function renderLoop(timestamp) {
        requestAnimationFrame(renderLoop);

        // Update OrbitControls damping
        controls.update();

        // Smoothly orient active mob heads to face camera
        orientEntityHeadsToCamera();

        // Pulsing animation for hostile entities and sightline laser tip
        const pulse = 0.5 + 0.5 * Math.sin(timestamp * 0.008);
        if (sightlineTip) {
            sightlineTip.scale.setScalar(0.9 + 0.3 * pulse);
        }

        // Render Three.js Scene
        renderer.render(scene, camera);
    }

    // Initialize application when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
