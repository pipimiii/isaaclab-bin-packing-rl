import argparse
import csv
import os
import sys
import time

# Benchmark-only arguments
# Environment / IsaacLab arguments are passed through to bin_packing_env.py.

benchmark_parser = argparse.ArgumentParser(
    description="Timing benchmark for IsaacLab bin packing environment",
    add_help=True,
)

benchmark_parser.add_argument(
    "--benchmark_out_dir",
    type=str,
    default="benchmark_results",
    help="Directory for benchmark CSV outputs",
)
benchmark_parser.add_argument(
    "--benchmark_name",
    type=str,
    default="bin_packing_timing",
    help="Name prefix for benchmark output files",
)
benchmark_parser.add_argument(
    "--benchmark_policy",
    type=str,
    default="random",
    choices=["random"],
    help="Policy used during benchmark. Currently only random is supported.",
)

benchmark_args, env_args = benchmark_parser.parse_known_args()

sys.argv = [sys.argv[0]] + env_args

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# bin_packing_env.py owns the IsaacLab AppLauncher.
# Importing it launches the simulation app and parses environment arguments.
import bin_packing_env as bpe  # type: ignore  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def make_env():
    """Create the IsaacLab simulation, scene, camera, and RL-style environment."""
    sim_cfg = bpe.sim_utils.SimulationCfg(device=bpe.args_cli.device)
    sim = bpe.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.0, -3.0, 5.0],
        target=[0.0, 0.0, 0.75],
    )

    scene_cfg = bpe.CubeDropSceneCfg(
        num_envs=bpe.args_cli.num_envs,
        env_spacing=bpe.args_cli.env_spacing,
    )
    scene = bpe.InteractiveScene(scene_cfg)

    camera = bpe.make_debug_camera(sim)

    sim.reset()

    camera.reset()
    camera.update(dt=sim.get_physics_dt())

    cam_pos = torch.tensor([[0.0, -2.6, 4.8]], device=sim.device)
    cam_target = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(cam_pos, cam_target)
    camera.update(dt=sim.get_physics_dt())

    sim.play()

    env = bpe.BinPackingRLEnv(sim=sim, scene=scene, camera=camera)
    return env


def tensor_mean_item(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        return x.float().mean().item()
    return float(x)


def tensor_sum_item(x, default=0):
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        return int(x.sum().item())
    return int(x)


def run_benchmark():
    os.makedirs(benchmark_args.benchmark_out_dir, exist_ok=True)

    episode_csv = os.path.join(
        benchmark_args.benchmark_out_dir,
        f"timing_{benchmark_args.benchmark_name}_episodes.csv",
    )
    summary_csv = os.path.join(
        benchmark_args.benchmark_out_dir,
        f"timing_{benchmark_args.benchmark_name}_summary.csv",
    )

    env = make_env()

    print("[INFO] Timing benchmark started", flush=True)
    print(f"[INFO] benchmark_name  = {benchmark_args.benchmark_name}", flush=True)
    print(f"[INFO] policy          = {benchmark_args.benchmark_policy}", flush=True)
    print(f"[INFO] num_episodes    = {bpe.args_cli.num_episodes}", flush=True)
    print(f"[INFO] num_envs        = {bpe.args_cli.num_envs}", flush=True)
    print(f"[INFO] num_cubes       = {bpe.args_cli.num_cubes}", flush=True)
    print(f"[INFO] cube_size       = {bpe.args_cli.cube_size}", flush=True)
    print(f"[INFO] obs_type        = {bpe.args_cli.obs_type}", flush=True)
    print(f"[INFO] reward_type     = {bpe.args_cli.reward_type}", flush=True)
    print(f"[INFO] episode CSV     = {episode_csv}", flush=True)
    print(f"[INFO] summary CSV     = {summary_csv}", flush=True)

    episode_rows = []

    with open(episode_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "num_envs",
            "num_cubes",
            "cube_size",
            "obs_type",
            "reward_type",
            "sim_steps",
            "sim_time_s",
            "wall_time_s",
            "fps",
            "real_time_factor",
            "success_envs",
            "timeout_envs",
            "reward_sum_mean",
            "avg_drop_cycle_sim_steps",
            "avg_drop_cycle_wall_time_s",
            "num_drop_cycles",
        ])

        for ep in range(bpe.args_cli.num_episodes):
            if not bpe.simulation_app.is_running():
                break

            obs = env.reset()
            del obs

            ep_wall_t0 = time.perf_counter()
            cycle_wall_times = []
            cycle_sim_steps = []

            final_info = None
            final_done = None

            while bpe.simulation_app.is_running():
                action = env.random_action()

                cycle_wall_t0 = time.perf_counter()
                obs, reward, done, info = env.step(action)
                cycle_wall_t1 = time.perf_counter()

                del obs

                cycle_wall_times.append(cycle_wall_t1 - cycle_wall_t0)
                cycle_sim_steps.append(int(info.get("settle_steps", 0)))

                final_info = info
                final_done = done

                if torch.all(done):
                    break

            ep_wall_time = time.perf_counter() - ep_wall_t0
            sim_steps = int(env.episode_step)
            sim_time = sim_steps * env.sim_dt
            fps = sim_steps / max(ep_wall_time, 1.0e-8)
            rtf = sim_time / max(ep_wall_time, 1.0e-8)

            if final_info is None:
                success_envs = 0
                timeout_envs = 0
            else:
                success_envs = tensor_sum_item(final_info.get("success_envs"))
                timeout_envs = tensor_sum_item(final_info.get("timeout_envs"))

            reward_sum_mean = tensor_mean_item(env.episode_reward_sum)

            avg_cycle_steps = float(np.mean(cycle_sim_steps)) if cycle_sim_steps else 0.0
            avg_cycle_wall = float(np.mean(cycle_wall_times)) if cycle_wall_times else 0.0
            num_cycles = len(cycle_sim_steps)

            row = {
                "episode": ep + 1,
                "num_envs": bpe.args_cli.num_envs,
                "num_cubes": bpe.args_cli.num_cubes,
                "cube_size": bpe.args_cli.cube_size,
                "obs_type": bpe.args_cli.obs_type,
                "reward_type": bpe.args_cli.reward_type,
                "sim_steps": sim_steps,
                "sim_time_s": sim_time,
                "wall_time_s": ep_wall_time,
                "fps": fps,
                "real_time_factor": rtf,
                "success_envs": success_envs,
                "timeout_envs": timeout_envs,
                "reward_sum_mean": reward_sum_mean,
                "avg_drop_cycle_sim_steps": avg_cycle_steps,
                "avg_drop_cycle_wall_time_s": avg_cycle_wall,
                "num_drop_cycles": num_cycles,
            }
            episode_rows.append(row)

            writer.writerow([
                row["episode"],
                row["num_envs"],
                row["num_cubes"],
                row["cube_size"],
                row["obs_type"],
                row["reward_type"],
                row["sim_steps"],
                row["sim_time_s"],
                row["wall_time_s"],
                row["fps"],
                row["real_time_factor"],
                row["success_envs"],
                row["timeout_envs"],
                row["reward_sum_mean"],
                row["avg_drop_cycle_sim_steps"],
                row["avg_drop_cycle_wall_time_s"],
                row["num_drop_cycles"],
            ])

            print(
                f"[EP {ep + 1:03d}/{bpe.args_cli.num_episodes}] "
                f"steps={sim_steps:05d} | "
                f"wall={ep_wall_time:.3f}s | "
                f"FPS={fps:.2f} | "
                f"RTF={rtf:.3f} | "
                f"success={success_envs}/{bpe.args_cli.num_envs} | "
                f"reward_mean={reward_sum_mean:.4f}",
                flush=True,
            )

    if not episode_rows:
        print("[WARN] No benchmark episodes were completed.", flush=True)
        bpe.simulation_app.close()
        return

    def mean_of(key):
        return float(np.mean([r[key] for r in episode_rows]))

    def std_of(key):
        return float(np.std([r[key] for r in episode_rows]))

    summary = {
        "num_episodes_completed": len(episode_rows),
        "num_envs": bpe.args_cli.num_envs,
        "num_cubes": bpe.args_cli.num_cubes,
        "cube_size": bpe.args_cli.cube_size,
        "obs_type": bpe.args_cli.obs_type,
        "reward_type": bpe.args_cli.reward_type,
        "mean_sim_steps": mean_of("sim_steps"),
        "std_sim_steps": std_of("sim_steps"),
        "mean_sim_time_s": mean_of("sim_time_s"),
        "mean_wall_time_s": mean_of("wall_time_s"),
        "std_wall_time_s": std_of("wall_time_s"),
        "mean_fps": mean_of("fps"),
        "std_fps": std_of("fps"),
        "mean_real_time_factor": mean_of("real_time_factor"),
        "std_real_time_factor": std_of("real_time_factor"),
        "mean_success_envs": mean_of("success_envs"),
        "mean_timeout_envs": mean_of("timeout_envs"),
        "mean_reward_sum": mean_of("reward_sum_mean"),
        "mean_drop_cycle_sim_steps": mean_of("avg_drop_cycle_sim_steps"),
        "mean_drop_cycle_wall_time_s": mean_of("avg_drop_cycle_wall_time_s"),
    }

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n[INFO] Timing benchmark completed", flush=True)
    print(f"[INFO] Episodes completed: {summary['num_episodes_completed']}", flush=True)
    print(f"[INFO] Mean sim steps:     {summary['mean_sim_steps']:.2f}", flush=True)
    print(f"[INFO] Mean wall time:     {summary['mean_wall_time_s']:.4f} s", flush=True)
    print(f"[INFO] Mean FPS:           {summary['mean_fps']:.2f}", flush=True)
    print(f"[INFO] Mean RTF:           {summary['mean_real_time_factor']:.3f}", flush=True)
    print(f"[INFO] Mean success envs:  {summary['mean_success_envs']:.2f}/{bpe.args_cli.num_envs}", flush=True)
    print(f"[INFO] Episode CSV:        {episode_csv}", flush=True)
    print(f"[INFO] Summary CSV:        {summary_csv}", flush=True)

    bpe.simulation_app.close()


if __name__ == "__main__":
    run_benchmark()