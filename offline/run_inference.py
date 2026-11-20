"""
Single-Inference Mode Runner Script for Squared Sense MC.

Loads a trained Actor-Critic policy checkpoint, executes single-episode pathfinding
rollouts with configurable playback speed (FPS) and stepping controls, and streams
live multi-LOD voxel observations along with rich Inference Telemetry directly into
the shared memory (MMF) bus for real-time inspection in the 3D WebGL visualizer.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import glob
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from offline.adapter import ObservationAdapter
from offline.ipc_emitter import MinecraftIpcEmitter
from offline.ipc_telemetry import InferenceTelemetry
from offline.lod_extractor import WorldVolume
from offline.palette import get_palette
from offline.pathfinding_env import (
    ACTION_NAMES,
    CARDINAL_VECTORS,
    MinecraftPathfindingEnv,
    MoveOutcome
)
from offline.spawn_indexer import SpawnSampler
from offline.train_ppo import VoxelActorCritic


def run_inference_episode(
    env: MinecraftPathfindingEnv,
    model: Optional[Any],
    emitter: MinecraftIpcEmitter,
    adapter: ObservationAdapter,
    checkpoint_meta: Dict[str, Any],
    fps: float = 5.0,
    greedy: bool = True,
    step_pause: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes a single pathfinding episode with live IPC streaming.
    """
    obs, info = env.reset()
    done = False
    step_count = 0
    total_return = 0.0

    target_pos = env.target_pos
    l_bfs = env.l_bfs
    optimal_path = env.optimal_path

    print(f"\n=======================================================")
    print(f"Episode Started! Spawn: {env.spawn_pos} -> Target: {target_pos}")
    print(f"Optimal BFS Length: {l_bfs} blocks | Heading: {env.cardinal}")
    print(f"Streaming live frames to 3D visualizer at {fps:.1f} FPS...")
    print(f"=======================================================")

    while not done:
        player_pos = env.player_pos
        curr_dist = env.prev_dist

        # 1. Adapt observation for model
        adapted = adapter.adapt(
            raw_obs=obs,
            player_pos=player_pos,
            target_pos=target_pos,
            cardinal=env.cardinal,
            camera_pitch=env.camera_pitch,
            prev_action=env.prev_action
        )

        action_mask = obs.get("action_mask", env.get_action_mask())
        action_probs = [1.0 / env.num_actions] * env.num_actions
        critic_val = 0.0
        chosen_action = 0

        # 2. Query Policy Model
        if model is not None and HAS_TORCH:
            with torch.no_grad():
                s_t = torch.from_numpy(adapted["shells"]).unsqueeze(0).to(device)
                st_t = torch.from_numpy(adapted["state"]).unsqueeze(0).to(device)
                g_t = torch.from_numpy(adapted["goal_vec"]).unsqueeze(0).to(device)
                m_t = torch.from_numpy(action_mask).unsqueeze(0).to(device)

                feat = model.forward_features(s_t, st_t, g_t)
                logits = model.actor(feat)
                # Apply action masking (safe fallback if all masked)
                mask_bool = m_t.bool()
                all_masked = ~mask_bool.any(dim=-1, keepdim=True)
                safe_mask = mask_bool | all_masked
                logits = torch.where(safe_mask, logits, torch.tensor(-1e9, device=device))
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                action_probs = [float(p) for p in probs]

                if greedy:
                    chosen_action = int(np.argmax(action_probs))
                else:
                    chosen_action = int(np.random.choice(len(action_probs), p=probs))

                critic_val = float(model.critic(feat).squeeze().item())
        else:
            # Fallback heuristic: choose action moving closer to target
            valid_actions = [a for a in range(env.num_actions) if action_mask[a]]
            chosen_action = valid_actions[0] if valid_actions else 0

        # 3. Pack Inference Telemetry Header (world deltas for 3D visualizer alignment)
        target_rel = (
            float(target_pos[0] - player_pos[0]),
            float(target_pos[1] - player_pos[1]),
            float(target_pos[2] - player_pos[2])
        )
        tel_bytes = InferenceTelemetry.pack(
            target_pos=target_pos,
            player_pos=player_pos,
            target_rel=target_rel,
            remaining_dist=curr_dist,
            step_reward=0.0,
            episode_return=total_return,
            step_count=step_count,
            critic_value=critic_val,
            action_probs=action_probs,
            chosen_action=chosen_action,
            l_bfs=l_bfs,
            visited_trail=env.visited_trail,
            bfs_path=optimal_path,
            checkpoint_steps=checkpoint_meta.get("steps", 0),
            checkpoint_policy_loss=checkpoint_meta.get("latest_policy_loss", 0.0),
            checkpoint_value_loss=checkpoint_meta.get("latest_value_loss", 0.0),
            checkpoint_success_rate=checkpoint_meta.get("latest_success_rate", 0.0)
        )

        # 4. Stream observation + telemetry to IPC bus
        full_obs = env.extractor.extract_observation(
            player_pos=player_pos,
            cardinal=env.cardinal,
            camera_pitch=env.camera_pitch,
            apply_visibility_culling=True
        )
        emitter.emit_observation(full_obs, game_tick=step_count, telemetry_bytes=tel_bytes)

        action_name = ACTION_NAMES[chosen_action] if chosen_action < len(ACTION_NAMES) else f"ACT_{chosen_action}"
        print(f"  [Step {step_count:02d}] Pos: {player_pos} | Dist: {curr_dist:.1f}m | "
              f"Action: {action_name:<10} (p={action_probs[chosen_action]:.2f}) | "
              f"Critic V(s): {critic_val:+.2f}")

        if step_pause:
            input("    [PAUSED] Press Enter to execute step...")
        elif fps > 0:
            time.sleep(1.0 / fps)

        # 5. Execute Step in Environment
        next_obs, reward, term, trunc, info = env.step(chosen_action)
        total_return += reward
        step_count += 1
        obs = next_obs
        done = term or trunc

    # Final frame emission
    final_target_rel = (
        float(target_pos[0] - env.player_pos[0]),
        float(target_pos[1] - env.player_pos[1]),
        float(target_pos[2] - env.player_pos[2])
    )
    final_tel = InferenceTelemetry.pack(
        target_pos=target_pos,
        player_pos=env.player_pos,
        target_rel=final_target_rel,
        remaining_dist=env.prev_dist,
        step_reward=reward,
        episode_return=total_return,
        step_count=step_count,
        critic_value=critic_val,
        action_probs=action_probs,
        chosen_action=chosen_action,
        l_bfs=l_bfs,
        visited_trail=env.visited_trail,
        bfs_path=optimal_path,
        checkpoint_steps=checkpoint_meta.get("steps", 0)
    )
    final_obs = env.extractor.extract_observation(
        player_pos=env.player_pos,
        cardinal=env.cardinal,
        camera_pitch=env.camera_pitch,
        apply_visibility_culling=True
    )
    emitter.emit_observation(final_obs, game_tick=step_count, telemetry_bytes=final_tel)

    is_success = bool(info.get("is_success", False))
    efficiency = float(info.get("efficiency_ratio", 1.0))

    print(f"\nEpisode Finished! Success: {is_success} | Outcome: {info.get('outcome')}")
    print(f"Total Return: {total_return:.2f} | Steps: {step_count}/{env.max_steps} | Path Efficiency: {efficiency:.2f}")

    return {
        "success": is_success,
        "steps": step_count,
        "l_bfs": l_bfs,
        "total_return": total_return,
        "efficiency": efficiency
    }


def main():
    parser = argparse.ArgumentParser(description="Squared Sense MC — Single-Inference Runner")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--region_dir", type=str, default="baked_regions_test", help="Path to baked region memmaps")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed in frames per second")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes to run")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum distance stage (0..3)")
    parser.add_argument("--pause", action="store_true", help="Pause for user confirmation between steps")
    parser.add_argument("--greedy", action="store_true", default=True, help="Select greedy argmax actions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # 1. Setup Environment
    palette = get_palette()
    world = WorldVolume(args.region_dir)
    spawns = glob.glob(os.path.join(args.region_dir, "*.spawns.npz"))
    if not spawns:
        print(f"Error: No .spawns.npz found in {args.region_dir}")
        return
    sampler = SpawnSampler.from_npz(spawns[0])

    env = MinecraftPathfindingEnv(
        world_volume=world,
        spawn_sampler=sampler,
        palette_obj=palette,
        allow_turns=True,
        mask_fatal=True
    )
    env.curriculum.current_stage = min(3, max(0, args.stage))

    # 2. Setup IPC Emitter & Adapter
    emitter = MinecraftIpcEmitter()
    adapter = ObservationAdapter(palette=palette)

    # 3. Load Checkpoint if specified
    model = None
    checkpoint_meta = {}
    if args.checkpoint and os.path.exists(args.checkpoint) and HAS_TORCH:
        print(f"Loading policy checkpoint from: {args.checkpoint}...")
        model = VoxelActorCritic(num_actions=env.num_actions)
        ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()

        meta_file = args.checkpoint.replace(".pt", ".json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                checkpoint_meta = json.load(f)
        print("Checkpoint loaded successfully!")

    # 4. Execute Episodes
    success_count = 0
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        res = run_inference_episode(
            env=env,
            model=model,
            emitter=emitter,
            adapter=adapter,
            checkpoint_meta=checkpoint_meta,
            fps=args.fps,
            greedy=args.greedy,
            step_pause=args.pause
        )
        if res["success"]:
            success_count += 1

    print(f"\n=======================================================")
    print(f"All {args.episodes} episodes completed! Success Rate: {success_count / args.episodes * 100:.1f}%")
    print(f"=======================================================\n")
    emitter.close()


if __name__ == "__main__":
    main()
