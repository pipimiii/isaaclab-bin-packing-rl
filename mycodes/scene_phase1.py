import argparse
import time

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-1 cube drop environment")

#parser.add_argument("--device", type=str, default="cuda", help="Device to run simulation on")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Environment spacing")

# Phase-1 task parameters
parser.add_argument("--num_cubes", type=int, default=5, help="Active cubes per episode (<= MAX_CUBES)")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--max_episode_steps", type=int, default=800, help="Timeout in physics steps")
# Project variants
parser.add_argument("--cube_size", type=float, default=0.05, help="Cube edge length in meters")
parser.add_argument(
    "--reward_type",
    type=str,
    default="sparse",
    choices=["sparse", "dense", "hybrid"],
    help="Reward variant: sparse, dense, or hybrid",
)
parser.add_argument(
    "--obs_type",
    type=str,
    default="vector",
    choices=["vector", "image", "multimodal"],
    help="Observation variant: vector, image, or multimodal",
)
#simulate parameter
parser.add_argument("--drop_interval_s", type=float, default=1.0, help="Simulated seconds between cube drops")
# Drop generation region, relative to each environment/bin origin
parser.add_argument("--drop_xy_half", type=float, default=0.08, help="Half-width of random x/y drop region above the bin")
parser.add_argument("--drop_z_min", type=float, default=0.35, help="Minimum local z height for cube drop")
parser.add_argument("--drop_z_max", type=float, default=0.50, help="Maximum local z height for cube drop")

#flag for RL 
parser.add_argument(
    "--use_action_drop",
    action="store_true",
    default=False,
    help="Use normalized action-style random drop instead of old scripted random drop",
)

# Sequential drop strategy for future RL baseline
parser.add_argument("--fixed_interval_drops", action="store_true", default=False, help="Use old fixed-time drop schedule instead of stability-based dropping")
parser.add_argument("--drop_wait_min_steps", type=int, default=20, help="Minimum physics steps to wait after each drop before checking pile stability")
parser.add_argument("--pile_stable_speed_thresh", type=float, default=0.08, help="Max cube speed threshold for pile stability before next drop")
parser.add_argument("--pile_stable_steps", type=int, default=20, help="Consecutive stable steps required before dropping the next cube")
parser.add_argument("--max_steps_between_drops", type=int, default=240, help="Force next cube after this many steps even if pile is not stable")
# Phase-1 success condition: "in bin and stable"
parser.add_argument("--bin_half_x", type=float, default=0.19, help="Half-width of in-bin region in x (meters)")
parser.add_argument("--bin_half_y", type=float, default=0.25, help="Half-width of in-bin region in y (meters)")
parser.add_argument("--bin_floor_z", type=float, default=0.015, help="Minimum z relative to env origin to count as inside bin")
parser.add_argument("--bin_top_z", type=float, default=0.30, help="Maximum z relative to env origin to count as inside bin")
parser.add_argument("--stable_speed_thresh", type=float, default=0.05, help="Linear speed threshold for stability (m/s)")
parser.add_argument("--stable_steps", type=int, default=15, help="Consecutive physics steps needed for stability")

#camera 
parser.add_argument("--save_camera", action="store_true", default=False, help="Save RGB camera frames")
parser.add_argument("--camera_interval", type=int, default=10, help="Save one frame every N physics steps")
parser.add_argument("--camera_out_dir", type=str, default="camera_output", help="Directory to save camera images")
# Isaac Lab app launcher args (includes --headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
import torch
import os
import numpy as np
from PIL import Image

from omni.isaac.lab.sensors.camera import Camera, CameraCfg
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCollection, RigidObjectCollectionCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext, schemas
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


# Constants
# 
MAX_CUBES = 40  

if args_cli.num_cubes < 1 or args_cli.num_cubes > MAX_CUBES:
    raise ValueError(f"--num_cubes must be in [1, {MAX_CUBES}], got {args_cli.num_cubes}")



# Scene config
# 
class CubeDropSceneCfg(InteractiveSceneCfg):
    """Phase-1 scene: ground, light, one KLT bin per env, and MAX_CUBES rigid cubes per env."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5)),
    )

    # Keep the original asset and scale.
    # Removed fixed_tendons_props because the KLT bin is just a static rigid bin for this task.
    bin = AssetBaseCfg(
    prim_path="/World/envs/env_.*/small_KLT",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.14636),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
        scale=(2, 2, 2),
        rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    ),
)
    object_collection = RigidObjectCollectionCfg(
        rigid_objects={
            f"cube_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Origin{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(args_cli.cube_size, args_cli.cube_size, args_cli.cube_size),  # keep your original cube size
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        metallic=0.2,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 2.0),
                ),
            )
            for i in range(MAX_CUBES)
        }
    )

    def __init__(self, *, num_envs: int = 2, env_spacing: float = 2.0, **kwargs):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing, **kwargs)

    def _build(self, simulator: SimulationContext):
        # Keep your original Origin{i} structure
        for env_idx in range(self.num_envs):
            for i in range(MAX_CUBES):
                prim_utils.create_prim(
                    f"/World/envs/env_{env_idx}/Origin{i}",
                    "Xform",
                    translation=[0.0, 0.0, 0.0],
                )
        super()._build(simulator)

def make_debug_camera(sim: SimulationContext):
    camera_cfg = CameraCfg(
        prim_path="/World/DebugCamera",
        update_period=0,
        height=1600,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,              # wider view than 24
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    return camera

def reset_episode(sim: SimulationContext, scene: InteractiveScene, num_cubes: int):
    """Reset all cubes, park inactive cubes far away, and clear scene buffers."""
    collection: RigidObjectCollection = scene["object_collection"]

    object_state = collection.data.default_object_state.clone()

    # Start from world-frame default state and shift by env origin, following Isaac Lab reset style.
    object_state[..., :3] += scene.env_origins.unsqueeze(1)

    num_envs = scene.num_envs
    device = object_state.device

    # Zero velocities for all cubes
    object_state[..., 7:] = 0.0

    # Park ALL cubes far away first so inactive cubes do not affect the episode.
    parked_offsets = []
    for i in range(MAX_CUBES):
        parked_offsets.append([50.0 + 0.25 * i, 0.0, 0.25])
    parked_offsets = torch.tensor(parked_offsets, device=device)  # [MAX_CUBES, 3]
    object_state[..., :3] = scene.env_origins.unsqueeze(1) + parked_offsets.unsqueeze(0)

    # Write full collection state to sim
    collection.write_object_link_pose_to_sim(object_state[..., :7])
    collection.write_object_com_velocity_to_sim(object_state[..., 7:])

    # Clear internal buffers after writing state
    scene.reset()
    sim.reset(soft=True)

    # Per-env / per-cube consecutive stable-step counters
    stable_counter = torch.zeros((num_envs, MAX_CUBES), dtype=torch.long, device=device)

    # Per-env done/success flags
    done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)

    return stable_counter, done_envs, success_envs

def drop_cube(scene: InteractiveScene, cube_idx: int):
    """Spawn one active cube above the bin using a configurable centered drop region."""
    collection: RigidObjectCollection = scene["object_collection"]

    ids, _ = collection.find_objects(f"cube_{cube_idx}", preserve_order=True)

    object_state = collection.data.default_object_state.clone()
    num_envs = scene.num_envs
    dev_str = str(object_state.device)

    # Controlled drop region above the bin.
    # This is better for packing many cubes than the old wide random region.
    xy = math_utils.sample_uniform(
        -args_cli.drop_xy_half,
        args_cli.drop_xy_half,
        size=(num_envs, 2),
        device=dev_str,
    )

    z = math_utils.sample_uniform(
        args_cli.drop_z_min,
        args_cli.drop_z_max,
        size=(num_envs, 1),
        device=dev_str,
    )

    drop_offset = torch.cat([xy, z], dim=1)
    drop_position = scene.env_origins + drop_offset

    object_state[:, cube_idx, :3] = drop_position

    object_state[:, cube_idx, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=object_state.device,
    ).view(1, 4).repeat(num_envs, 1)

    object_state[:, cube_idx, 7:] = 0.0

    collection.write_object_link_pose_to_sim(
        object_state[:, cube_idx, :7].unsqueeze(1),
        object_ids=ids,
    )

    collection.write_object_com_velocity_to_sim(
        object_state[:, cube_idx, 7:].unsqueeze(1),
        object_ids=ids,
    )

def drop_cube_at_action(scene: InteractiveScene, cube_idx: int, action_xy: torch.Tensor):
    """Drop one cube using an RL-style normalized action.

    Args:
        scene: Isaac Lab interactive scene.
        cube_idx: Which cube to drop.
        action_xy: Tensor of shape [num_envs, 2].
                   Each value should be in [-1, 1].
                   action_xy[:, 0] controls local x release position.
                   action_xy[:, 1] controls local y release position.

    Mapping:
        action -1 means left/back side of allowed drop region.
        action +1 means right/front side of allowed drop region.
        physical xy = action_xy * args_cli.drop_xy_half

    This is the RL replacement for the random xy sampling in drop_cube().
    """
    collection: RigidObjectCollection = scene["object_collection"]

    ids, _ = collection.find_objects(f"cube_{cube_idx}", preserve_order=True)

    object_state = collection.data.default_object_state.clone()
    num_envs = scene.num_envs
    device = object_state.device

    # Make sure action is a tensor on the correct device.
    action_xy = action_xy.to(device=device, dtype=torch.float32)

    if action_xy.shape != (num_envs, 2):
        raise ValueError(
            f"action_xy must have shape ({num_envs}, 2), got {tuple(action_xy.shape)}"
        )

    # Clamp to valid normalized action range.
    action_xy = torch.clamp(action_xy, -1.0, 1.0)

    # Convert normalized action to local physical drop offset.
    xy = action_xy * args_cli.drop_xy_half

    # Use fixed z for RL. This keeps the action space simple: action = [x, y].
    z = torch.full(
        (num_envs, 1),
        fill_value=args_cli.drop_z_max,
        device=device,
        dtype=torch.float32,
    )

    drop_offset = torch.cat([xy, z], dim=1)
    drop_position = scene.env_origins + drop_offset

    object_state[:, cube_idx, :3] = drop_position

    # No rotation for now. Later we can add yaw as a third action.
    object_state[:, cube_idx, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=device,
        dtype=torch.float32,
    ).view(1, 4).repeat(num_envs, 1)

    # Start from zero velocity.
    object_state[:, cube_idx, 7:] = 0.0

    collection.write_object_link_pose_to_sim(
        object_state[:, cube_idx, :7].unsqueeze(1),
        object_ids=ids,
    )

    collection.write_object_com_velocity_to_sim(
        object_state[:, cube_idx, 7:].unsqueeze(1),
        object_ids=ids,
    )

def get_vector_obs(scene: InteractiveScene, num_cubes: int, dropped_cubes: int):
    """Return vector observation for RL.

    Observation contains:
    - cube local positions
    - cube linear velocities
    - active cube mask
    - dropped cube mask
    - current progress information

    Shape:
        [num_envs, obs_dim]

    This is the simplest observation type and should be the first one used for PPO.
    """
    collection: RigidObjectCollection = scene["object_collection"]

    state_w = collection.data.object_link_state_w
    device = state_w.device
    num_envs = scene.num_envs

    # Use all MAX_CUBES so the observation size stays fixed.
    pos_w = state_w[:, :MAX_CUBES, :3]
    lin_vel_w = state_w[:, :MAX_CUBES, 7:10]

    # Convert world position to local position relative to each bin/env origin.
    pos_local = pos_w - scene.env_origins.unsqueeze(1)

    # Masks.
    cube_ids = torch.arange(MAX_CUBES, device=device).view(1, MAX_CUBES)

    active_mask = cube_ids < num_cubes
    dropped_mask = cube_ids < dropped_cubes

    active_mask_f = active_mask.float().repeat(num_envs, 1)
    dropped_mask_f = dropped_mask.float().repeat(num_envs, 1)

    # Zero out inactive cubes so parked cubes far away do not pollute observation.
    valid_mask = active_mask_f.unsqueeze(-1)

    pos_local = pos_local * valid_mask
    lin_vel_w = lin_vel_w * valid_mask

    # Normalize positions approximately by bin dimensions.
    # This helps RL because values become order 1 instead of raw meters.
    pos_scale = torch.tensor(
        [args_cli.bin_half_x, args_cli.bin_half_y, args_cli.bin_top_z],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3)

    pos_norm = pos_local / torch.clamp(pos_scale, min=1.0e-6)

    # Velocity scale: 1 m/s reference. Keep simple.
    vel_norm = lin_vel_w

    progress = torch.tensor(
        [dropped_cubes / max(num_cubes, 1), num_cubes / MAX_CUBES],
        device=device,
        dtype=torch.float32,
    ).view(1, 2).repeat(num_envs, 1)

    obs = torch.cat(
        [
            pos_norm.reshape(num_envs, -1),
            vel_norm.reshape(num_envs, -1),
            active_mask_f,
            dropped_mask_f,
            progress,
        ],
        dim=1,
    )

    return obs

def get_image_obs(camera: Camera | None):
    """Return RGB image observation from the debug camera.

    Output:
        image tensor normalized to [0, 1].

    Note:
        current camera is a single global debug camera.
        For a real multi-env image RL setup, we later create one camera per env.
        For the project requirement, this function provides a selectable image observation path.
    """
    if camera is None:
        raise ValueError("get_image_obs() requested but camera is None.")

    rgb = camera.data.output["rgb"][..., :3]

    # Convert to float in [0, 1].
    rgb = rgb.float() / 255.0

    return rgb

def get_obs(
    scene: InteractiveScene,
    camera: Camera | None,
    num_cubes: int,
    dropped_cubes: int,
):
    """Return observation according to args_cli.obs_type.

    Supported:
        vector
        image
        multimodal
    """
    if args_cli.obs_type == "vector":
        return get_vector_obs(scene, num_cubes, dropped_cubes)

    elif args_cli.obs_type == "image":
        return get_image_obs(camera)

    elif args_cli.obs_type == "multimodal":
        return {
            "vector": get_vector_obs(scene, num_cubes, dropped_cubes),
            "image": get_image_obs(camera),
        }

    else:
        raise ValueError(f"Unknown obs_type: {args_cli.obs_type}")

def compute_reward(
    cube_success_mask: torch.Tensor,
    prev_cubes_success_count: torch.Tensor,
    cubes_success_count: torch.Tensor,
    pos_local: torch.Tensor,
    speed: torch.Tensor,
    success_envs: torch.Tensor,
    dropped_cubes: int,
):
    """Compute reward according to selected reward type.

    Args:
        cube_success_mask:
            [num_envs, active_cubes] bool tensor.
        prev_cubes_success_count:
            [num_envs] previous number of successful cubes.
        cubes_success_count:
            [num_envs] current number of successful cubes.
        pos_local:
            [num_envs, active_cubes, 3] local cube positions.
        speed:
            [num_envs, active_cubes] cube speeds.
        success_envs:
            [num_envs] whether all active cubes are successful.
        dropped_cubes:
            number of cubes already dropped.

    Returns:
        reward:
            [num_envs] reward tensor.
    """
    device = cubes_success_count.device

    # New cube success since previous step.
    newly_successful = (cubes_success_count - prev_cubes_success_count).clamp(min=0).float()

    if dropped_cubes <= 0 or pos_local.numel() == 0:
        return torch.zeros_like(newly_successful, dtype=torch.float32)

    # Use currently active cubes only.
    active_pos = pos_local[:, :dropped_cubes, :]
    active_speed = speed[:, :dropped_cubes]

    # Check in-bin again for dense shaping.
    in_x = torch.abs(active_pos[..., 0]) <= args_cli.bin_half_x
    in_y = torch.abs(active_pos[..., 1]) <= args_cli.bin_half_y
    in_z = (active_pos[..., 2] >= args_cli.bin_floor_z) & (active_pos[..., 2] <= args_cli.bin_top_z)
    in_bin = in_x & in_y & in_z

    # Dense components.
    # 1. Encourage cubes to be inside the bin.
    inside_fraction = in_bin.float().mean(dim=1)

    # 2. Encourage compact placement near bin center in xy.
    xy_dist = torch.linalg.norm(active_pos[..., :2], dim=-1)
    xy_ref = max(args_cli.bin_half_x, args_cli.bin_half_y)
    xy_penalty = (xy_dist / max(xy_ref, 1.0e-6)).mean(dim=1)

    # 3. Encourage low pile height.
    height_ref = max(args_cli.bin_top_z, 1.0e-6)
    height_penalty = (active_pos[..., 2].clamp(min=0.0) / height_ref).mean(dim=1)

    # 4. Encourage stability.
    speed_penalty = active_speed.mean(dim=1)

    dense_reward = (
        0.5 * inside_fraction
        - 0.2 * xy_penalty
        - 0.1 * height_penalty
        - 0.1 * speed_penalty
    )

    # Sparse reward: only reward newly successful cubes.
    sparse_reward = newly_successful

    # Episode completion bonus.
    success_bonus = success_envs.float() * float(args_cli.num_cubes)

    if args_cli.reward_type == "sparse":
        reward = sparse_reward + success_bonus

    elif args_cli.reward_type == "dense":
        reward = dense_reward

    elif args_cli.reward_type == "hybrid":
        reward = sparse_reward + dense_reward + success_bonus

    else:
        raise ValueError(f"Unknown reward_type: {args_cli.reward_type}")

    return reward

def simulate_until_stable(
    sim: SimulationContext,
    scene: InteractiveScene,
    active_cubes: int,
    max_steps: int | None = None,
    camera: Camera | None = None,
):
    """Step physics until the current pile becomes stable or max_steps is reached.

    This is useful for RL because one RL action should mean:
        choose drop position -> simulate falling/settling -> return next observation.

    Returns:
        steps_used: number of physics steps simulated
        final_stable: bool tensor [num_envs]
        max_speed_per_env: tensor [num_envs]
    """
    if max_steps is None:
        max_steps = args_cli.max_steps_between_drops

    sim_dt = sim.get_physics_dt()
    device = scene.env_origins.device

    stable_counter = torch.zeros(scene.num_envs, dtype=torch.long, device=device)
    final_stable = torch.zeros(scene.num_envs, dtype=torch.bool, device=device)
    max_speed_per_env = torch.zeros(scene.num_envs, dtype=torch.float32, device=device)

    steps_used = 0

    for _ in range(max_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if camera is not None:
            camera.update(dt=sim_dt)

        steps_used += 1

        pile_stable_now, max_speed_per_env = check_pile_stable_for_next_drop(
            scene=scene,
            active_cubes=active_cubes,
            speed_thresh=args_cli.pile_stable_speed_thresh,
        )

        stable_counter = torch.where(
            pile_stable_now,
            stable_counter + 1,
            torch.zeros_like(stable_counter),
        )

        final_stable = stable_counter >= args_cli.pile_stable_steps

        if torch.all(final_stable):
            break

    return steps_used, final_stable, max_speed_per_env

def update_success_mask(
    scene: InteractiveScene,
    num_cubes: int,
    stable_counter: torch.Tensor,
    bin_half_x: float,
    bin_half_y: float,
    bin_floor_z: float,
    bin_top_z: float,
    stable_speed_thresh: float,
    stable_steps: int,
):
    """Return per-env success mask, per-cube success mask, and diagnostic info."""
    collection: RigidObjectCollection = scene["object_collection"]

    state_w = collection.data.object_link_state_w

    # Only evaluate active cubes [0:num_cubes]
    pos_w = state_w[:, :num_cubes, :3]
    lin_vel_w = state_w[:, :num_cubes, 7:10]

    # Local coordinates relative to each environment/bin origin
    pos_local = pos_w - scene.env_origins.unsqueeze(1)

    in_x = torch.abs(pos_local[..., 0]) <= bin_half_x
    in_y = torch.abs(pos_local[..., 1]) <= bin_half_y
    in_z = (pos_local[..., 2] >= bin_floor_z) & (pos_local[..., 2] <= bin_top_z)

    speed = torch.linalg.norm(lin_vel_w, dim=-1)
    stable_now = speed <= stable_speed_thresh

    in_bin_now = in_x & in_y & in_z
    good_now = in_bin_now & stable_now

    stable_counter[:, :num_cubes] = torch.where(
        good_now,
        stable_counter[:, :num_cubes] + 1,
        torch.zeros_like(stable_counter[:, :num_cubes]),
    )

    cube_success_mask = stable_counter[:, :num_cubes] >= stable_steps
    cubes_success_count = cube_success_mask.sum(dim=1)

    # Environment success means every active cube succeeded
    success_envs = torch.all(cube_success_mask, dim=1)

    return success_envs, stable_counter, cube_success_mask, cubes_success_count, good_now, pos_local, speed

def check_pile_stable_for_next_drop(
    scene: InteractiveScene,
    active_cubes: int,
    speed_thresh: float,
):
    """Check whether the currently dropped pile is stable enough for the next cube.

    Args:
        scene: InteractiveScene.
        active_cubes: Number of cubes already dropped.
        speed_thresh: Maximum allowed speed for each cube.

    Returns:
        pile_stable_now: [num_envs] bool tensor.
        max_speed_per_env: [num_envs] tensor.
    """
    device = scene.env_origins.device

    if active_cubes <= 0:
        return (
            torch.ones(scene.num_envs, dtype=torch.bool, device=device),
            torch.zeros(scene.num_envs, dtype=torch.float32, device=device),
        )

    collection: RigidObjectCollection = scene["object_collection"]
    state_w = collection.data.object_link_state_w

    lin_vel_w = state_w[:, :active_cubes, 7:10]
    speed = torch.linalg.norm(lin_vel_w, dim=-1)  # [num_envs, active_cubes]

    max_speed_per_env = speed.max(dim=1).values
    pile_stable_now = max_speed_per_env <= speed_thresh

    return pile_stable_now, max_speed_per_env

def run_phase1_episodes(sim: SimulationContext, scene: InteractiveScene, camera: Camera | None = None):
    """Run Phase-1 episodes with stability-based sequential dropping.

    Strategy:
    - Pre-create all cubes.
    - Reset parks all cubes away.
    - Drop cube_0 immediately.
    - For cube_i where i > 0, wait until the currently dropped pile is stable.
    - If pile never becomes stable, force the next cube after max_steps_between_drops.
    - Final success requires all active cubes to be inside the bin and stable.
    """
    collection: RigidObjectCollection = scene["object_collection"]
    sim_dt = sim.get_physics_dt()
    drop_interval_steps = max(1, int(round(args_cli.drop_interval_s / sim_dt)))

    print(f"[INFO] sim_dt = {sim_dt:.6f} s")
    print(f"[INFO] fixed_interval_drops = {args_cli.fixed_interval_drops}")
    print(f"[INFO] drop_interval_steps = {drop_interval_steps}")
    print(f"[INFO] num_cubes = {args_cli.num_cubes}")
    print(f"[INFO] max_episode_steps = {args_cli.max_episode_steps}")
    print(f"[INFO] drop_xy_half = {args_cli.drop_xy_half}")
    print(f"[INFO] drop_z_min/max = {args_cli.drop_z_min}, {args_cli.drop_z_max}")
    print(f"[INFO] pile_stable_speed_thresh = {args_cli.pile_stable_speed_thresh}")
    print(f"[INFO] pile_stable_steps = {args_cli.pile_stable_steps}")
    print(f"[INFO] max_steps_between_drops = {args_cli.max_steps_between_drops}")

    sim.play()

    for episode_idx in range(args_cli.num_episodes):
        print(f"\n{'=' * 80}")
        print(f"[INFO] Starting episode {episode_idx + 1}/{args_cli.num_episodes}")

        stable_counter, done_envs, success_envs = reset_episode(
            sim=sim,
            scene=scene,
            num_cubes=args_cli.num_cubes,
        )

        episode_step = 0
        dropped_cubes = 0

        # Time bookkeeping
        episode_real_t0 = time.perf_counter()
        cube_drop_sim_times = []
        cube_drop_real_times = []

        # New stability-based drop bookkeeping
        steps_since_last_drop = 0
        pile_stable_counter = torch.zeros(scene.num_envs, dtype=torch.long, device=scene.env_origins.device)

        # Per-cube success diagnostics
        cube_success_mask = torch.zeros(
            (scene.num_envs, args_cli.num_cubes),
            dtype=torch.bool,
            device=scene.env_origins.device,
        )
        cubes_success_count = torch.zeros(
            scene.num_envs,
            dtype=torch.long,
            device=scene.env_origins.device,
        )
        prev_cubes_success_count = torch.zeros(
        scene.num_envs,
        dtype=torch.long,
        device=scene.env_origins.device,
        )
        episode_reward_sum = torch.zeros(
        scene.num_envs,
        dtype=torch.float32,
        device=scene.env_origins.device,
        )

        pos_local_dbg = torch.zeros(
            (scene.num_envs, args_cli.num_cubes, 3),
            dtype=torch.float32,
            device=scene.env_origins.device,
        )
        speed_dbg = torch.zeros(
            (scene.num_envs, args_cli.num_cubes),
            dtype=torch.float32,
            device=scene.env_origins.device,
        )

        while simulation_app.is_running():
            
            # Decide whether to drop the next cube
            should_drop = False
            drop_reason = "none"

            if dropped_cubes < args_cli.num_cubes:
                if dropped_cubes == 0:
                    # Always start episode by dropping the first cube
                    should_drop = True
                    drop_reason = "first_cube"

                elif args_cli.fixed_interval_drops:
                    # Old behavior: fixed interval in simulation time
                    should_drop = (episode_step % drop_interval_steps == 0)
                    drop_reason = "fixed_interval"

                else:
                    # New behavior: wait until the current pile is stable
                    pile_stable_now, max_speed_per_env = check_pile_stable_for_next_drop(
                        scene=scene,
                        active_cubes=dropped_cubes,
                        speed_thresh=args_cli.pile_stable_speed_thresh,
                    )

                    pile_stable_counter = torch.where(
                        pile_stable_now,
                        pile_stable_counter + 1,
                        torch.zeros_like(pile_stable_counter),
                    )

                    waited_minimum = steps_since_last_drop >= args_cli.drop_wait_min_steps
                    all_envs_pile_stable = torch.all(pile_stable_counter >= args_cli.pile_stable_steps)
                    waited_too_long = steps_since_last_drop >= args_cli.max_steps_between_drops

                    if waited_minimum and all_envs_pile_stable:
                        should_drop = True
                        drop_reason = "pile_stable"

                    elif waited_too_long:
                        should_drop = True
                        drop_reason = "max_wait"

           
            # Drop cube if condition is satisfied
            if should_drop and dropped_cubes < args_cli.num_cubes:
                current_sim_time = episode_step * sim_dt
                current_real_time = time.perf_counter() - episode_real_t0

                if args_cli.use_action_drop:
                    # RL-style test path:
                    # create a random normalized action in [-1, 1]^2 for each environment.
                    # Later, PPO will replace this random_action_xy with the policy output.
                    random_action_xy = (
                        torch.rand(scene.num_envs, 2, device=scene.env_origins.device) * 2.0 - 1.0
                    )

                    drop_cube_at_action(
                        scene=scene,
                        cube_idx=dropped_cubes,
                        action_xy=random_action_xy,
                    )

                    action_info = f"action_xy={random_action_xy.detach().cpu().numpy()}"

                else:
                    # Old scripted version:
                    # this keeps your original behavior unchanged.
                    drop_cube(scene, cube_idx=dropped_cubes)

                    action_info = "old_random_drop"

                cube_drop_sim_times.append(current_sim_time)
                cube_drop_real_times.append(current_real_time)

                print(
                    f"[DROP ] cube_{dropped_cubes} | "
                    f"reason = {drop_reason} | "
                    f"mode = {'action_drop' if args_cli.use_action_drop else 'old_drop'} | "
                    f"episode_step = {episode_step} | "
                    f"episode_sim_time = {current_sim_time:.4f} s | "
                    f"episode_real_time = {current_real_time:.4f} s | "
                    f"steps_since_last_drop = {steps_since_last_drop} | "
                    f"{action_info}"
                )

                dropped_cubes += 1
                steps_since_last_drop = 0
                pile_stable_counter.zero_()

           
            # Isaac Lab simulation step
            
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            episode_step += 1
            steps_since_last_drop += 1

            
            # Camera update, observation debug print, and optional frame saving

            if camera is not None:
                camera.update(dt=sim_dt)

                # Debug observation shape print.
                # This runs only once near the beginning of each episode, so it will not spam output.
                if episode_step == 1:
                    obs = get_obs(
                        scene=scene,
                        camera=camera,
                        num_cubes=args_cli.num_cubes,
                        dropped_cubes=dropped_cubes,
                    )

                    if isinstance(obs, dict):
                        print("[OBS ] multimodal observation:")
                        print(f"       vector shape = {obs['vector'].shape}")
                        print(f"       image shape  = {obs['image'].shape}")
                    else:
                        print(f"[OBS ] {args_cli.obs_type} observation shape = {obs.shape}")

                if args_cli.save_camera and (episode_step % args_cli.camera_interval == 0):
                    os.makedirs(args_cli.camera_out_dir, exist_ok=True)

                    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                    img = Image.fromarray(rgb)
                    img.save(
                        os.path.join(
                            args_cli.camera_out_dir,
                            f"ep{episode_idx+1:03d}_step{episode_step:05d}.png",
                        )
                    )

            
            # Update success statistics for all cubes that have been dropped
            if dropped_cubes > 0:
                (
                    active_success_envs,
                    stable_counter,
                    active_cube_success_mask,
                    active_cubes_success_count,
                    good_now,
                    active_pos_local,
                    active_speed,
                ) = update_success_mask(
                    scene=scene,
                    num_cubes=dropped_cubes,
                    stable_counter=stable_counter,
                    bin_half_x=args_cli.bin_half_x,
                    bin_half_y=args_cli.bin_half_y,
                    bin_floor_z=args_cli.bin_floor_z,
                    bin_top_z=args_cli.bin_top_z,
                    stable_speed_thresh=args_cli.stable_speed_thresh,
                    stable_steps=args_cli.stable_steps,
                )

                cube_success_mask[:, :dropped_cubes] = active_cube_success_mask
                cubes_success_count = active_cubes_success_count

                pos_local_dbg[:, :dropped_cubes, :] = active_pos_local
                speed_dbg[:, :dropped_cubes] = active_speed

                reward = compute_reward(
                    cube_success_mask=active_cube_success_mask,
                    prev_cubes_success_count=prev_cubes_success_count,
                    cubes_success_count=active_cubes_success_count,
                    pos_local=active_pos_local,
                    speed=active_speed,
                    success_envs=active_success_envs,
                    dropped_cubes=dropped_cubes,
                )

                episode_reward_sum += reward
                prev_cubes_success_count = active_cubes_success_count.clone()

                # Only allow full episode success after all requested cubes have been dropped
                if dropped_cubes == args_cli.num_cubes:
                    success_envs = active_success_envs


            
            # Termination logic
            
            timed_out = episode_step >= args_cli.max_episode_steps
            timeout_envs = (~success_envs) & torch.full_like(success_envs, timed_out)
            done_envs = success_envs | timeout_envs

            if torch.all(done_envs):
                episode_real_t1 = time.perf_counter()
                episode_real_time = episode_real_t1 - episode_real_t0
                episode_sim_time = episode_step * sim_dt

                num_success = int(success_envs.sum().item())
                num_timeout = int(timeout_envs.sum().item())

                #save camera picture for the last step
                if camera is not None and args_cli.save_camera:
                    os.makedirs(args_cli.camera_out_dir, exist_ok=True)

                    camera.update(dt=sim_dt)
                    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                    img = Image.fromarray(rgb)
                    img.save(
                        os.path.join(
                            args_cli.camera_out_dir,
                            f"ep{episode_idx+1:03d}_final_step{episode_step:05d}.png",
                        )
                    )

                print(f"[INFO] Episode finished at step {episode_step}")
                print(f"[INFO] Episode simulated time: {episode_sim_time:.4f} s")
                print(f"[INFO] Episode real time:      {episode_real_time:.4f} s")
                print(f"[INFO] Success envs: {num_success}/{scene.num_envs}")
                print(f"[INFO] Timeout envs: {num_timeout}/{scene.num_envs}")
                print(f"[INFO] Reward type: {args_cli.reward_type}")
                print(f"[INFO] Episode reward sum per env:\n{episode_reward_sum}")  

                # Print per-cube drop intervals
                for i in range(len(cube_drop_sim_times)):
                    if i == 0:
                        sim_gap = cube_drop_sim_times[i]
                        real_gap = cube_drop_real_times[i]
                    else:
                        sim_gap = cube_drop_sim_times[i] - cube_drop_sim_times[i - 1]
                        real_gap = cube_drop_real_times[i] - cube_drop_real_times[i - 1]

                    print(
                        f"[TIME ] cube_{i} | "
                        f"drop_sim_time = {cube_drop_sim_times[i]:.4f} s | "
                        f"drop_real_time = {cube_drop_real_times[i]:.4f} s | "
                        f"since_prev_drop_sim = {sim_gap:.4f} s | "
                        f"since_prev_drop_real = {real_gap:.4f} s"
                    )

                # Per-environment cube success summary
                print("[INFO] Cubes succeeded per env:")
                for env_id in range(scene.num_envs):
                    print(
                        f"  env_{env_id}: "
                        f"{int(cubes_success_count[env_id].item())}/{args_cli.num_cubes} cubes succeeded | "
                        f"mask={cube_success_mask[env_id].detach().cpu().numpy()}"
                    )

                final_pos = collection.data.object_link_state_w[:, :args_cli.num_cubes, :3]
                final_pos_local = final_pos - scene.env_origins.unsqueeze(1)

                print(f"[INFO] Final active cube WORLD positions:\n{final_pos}")
                print(f"[INFO] Final active cube LOCAL positions:\n{final_pos_local}")
                print(f"[INFO] Final active cube speeds:\n{speed_dbg[:, :args_cli.num_cubes]}")

                break


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.0, -3.0, 5.0],
        target=[0.0, 0.0, 0.75],
    )

    scene_cfg = CubeDropSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
    )
    scene = InteractiveScene(scene_cfg)

    # create camera BEFORE reset
    camera = make_debug_camera(sim)

    # now initialize sim/scene
    sim.reset()

    # initialize camera internals
    camera.reset()
    print(f"[INFO] Camera initialized? {camera.is_initialized}")
    camera.update(dt=sim.get_physics_dt())

    # Camera above the center of all envs
    cam_pos = torch.tensor([[0.0, -2.6, 4.8]], device=sim.device)
    cam_target = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)

    camera.set_world_poses_from_view(cam_pos, cam_target)
    camera.update(dt=sim.get_physics_dt())

    print("[INFO] Setup complete...")

    run_phase1_episodes(sim, scene, camera=camera)

    simulation_app.close()

if __name__ == "__main__":
    main()