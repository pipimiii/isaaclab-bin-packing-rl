import argparse
import os
import time
from dataclasses import dataclass

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="RL-style IsaacLab bin packing environment")

# Environment size
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Spacing between environments")

# Task
parser.add_argument("--num_cubes", type=int, default=5, help="Active cubes per episode")
parser.add_argument("--max_cubes", type=int, default=40, help="Maximum cube assets created per env")
parser.add_argument("--cube_size", type=float, default=0.05, help="Cube edge length in meters")
parser.add_argument("--max_episode_steps", type=int, default=2000, help="Timeout in physics steps")

# Action/drop parameters
parser.add_argument("--drop_xy_half", type=float, default=0.12, help="Half-width of normalized action drop region")
parser.add_argument("--drop_z", type=float, default=0.50, help="Fixed local z release height")
parser.add_argument("--settle_max_steps", type=int, default=240, help="Max physics steps after each drop")
parser.add_argument(
    "--final_settle_max_steps",
    type=int,
    default=800,
    help="Max physics steps after the final cube is dropped",
)
parser.add_argument("--drop_wait_min_steps", type=int, default=20, help="Minimum steps to simulate after each drop")
parser.add_argument("--pile_stable_speed_thresh", type=float, default=0.08, help="Speed threshold for pile stability")
parser.add_argument("--pile_stable_steps", type=int, default=20, help="Consecutive stable steps for pile stability")

# Success definition
parser.add_argument("--bin_half_x", type=float, default=0.19, help="Half-width of in-bin region in x")
parser.add_argument("--bin_half_y", type=float, default=0.25, help="Half-width of in-bin region in y")
parser.add_argument("--bin_floor_z", type=float, default=0.015, help="Minimum local z to count as inside bin")
parser.add_argument("--bin_top_z", type=float, default=0.30, help="Maximum local z to count as inside bin")
parser.add_argument("--stable_speed_thresh", type=float, default=0.05, help="Cube speed threshold for success")
parser.add_argument("--stable_steps", type=int, default=15, help="Consecutive stable steps required for cube success")

# Reward / observation variants
parser.add_argument(
    "--reward_type",
    type=str,
    default="hybrid",
    choices=["sparse", "dense", "hybrid"],
    help="Reward variant",
)
parser.add_argument(
    "--obs_type",
    type=str,
    default="vector",
    choices=["vector", "image", "multimodal"],
    help="Observation variant",
)

# Direct random rollout test
parser.add_argument("--num_episodes", type=int, default=1, help="Number of random-policy test episodes")
parser.add_argument("--random_policy", action="store_true", default=True, help="Run random policy rollout test")

# Camera
parser.add_argument("--camera_height", type=int, default=1600, help="Camera image height")
parser.add_argument("--camera_width", type=int, default=1280, help="Camera image width")
parser.add_argument("--save_camera", action="store_true", default=False, help="Save camera frames")
parser.add_argument("--camera_interval", type=int, default=100, help="Save one camera image every N env steps")
parser.add_argument("--camera_out_dir", type=str, default="camera_output", help="Camera output directory")

# Isaac Lab app launcher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import numpy as np
import torch
from PIL import Image

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sim import SimulationContext, schemas
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR



# Validation
MAX_CUBES = args_cli.max_cubes

if args_cli.num_cubes < 1 or args_cli.num_cubes > MAX_CUBES:
    raise ValueError(f"--num_cubes must be in [1, {MAX_CUBES}], got {args_cli.num_cubes}")



# Scene config

class CubeDropSceneCfg(InteractiveSceneCfg):
    """Scene with ground, light, one static KLT bin per env, and MAX_CUBES cubes."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5)),
    )

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
                    size=(args_cli.cube_size, args_cli.cube_size, args_cli.cube_size),
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

    def __init__(self, *, num_envs: int, env_spacing: float, **kwargs):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing, **kwargs)

    def _build(self, simulator: SimulationContext):
        for env_idx in range(self.num_envs):
            for i in range(MAX_CUBES):
                prim_utils.create_prim(
                    f"/World/envs/env_{env_idx}/Origin{i}",
                    "Xform",
                    translation=[0.0, 0.0, 0.0],
                )
        super()._build(simulator)



# Camera
def make_debug_camera(sim: SimulationContext) -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/DebugCamera",
        update_period=0,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    return camera


# Environment helper functions
def reset_episode(sim: SimulationContext, scene: InteractiveScene):
    """Park all cubes far away and clear per-episode counters."""
    collection: RigidObjectCollection = scene["object_collection"]

    object_state = collection.data.default_object_state.clone()
    object_state[..., :3] += scene.env_origins.unsqueeze(1)
    object_state[..., 7:] = 0.0

    device = object_state.device
    num_envs = scene.num_envs

    parked_offsets = torch.tensor(
        [[50.0 + 0.25 * i, 0.0, 0.25] for i in range(MAX_CUBES)],
        device=device,
        dtype=torch.float32,
    )
    object_state[..., :3] = scene.env_origins.unsqueeze(1) + parked_offsets.unsqueeze(0)

    collection.write_object_link_pose_to_sim(object_state[..., :7])
    collection.write_object_com_velocity_to_sim(object_state[..., 7:])

    scene.reset()
    sim.reset(soft=True)

    stable_counter = torch.zeros((num_envs, MAX_CUBES), dtype=torch.long, device=device)
    success_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)

    return stable_counter, success_envs, done_envs


def drop_cube_at_action(scene: InteractiveScene, cube_idx: int, action_xy: torch.Tensor):
    """Drop cube using normalized action_xy in [-1, 1]^2."""
    collection: RigidObjectCollection = scene["object_collection"]

    ids, _ = collection.find_objects(f"cube_{cube_idx}", preserve_order=True)

    object_state = collection.data.default_object_state.clone()
    device = object_state.device
    num_envs = scene.num_envs

    action_xy = action_xy.to(device=device, dtype=torch.float32)

    if action_xy.shape != (num_envs, 2):
        raise ValueError(f"action_xy must have shape ({num_envs}, 2), got {tuple(action_xy.shape)}")

    action_xy = torch.clamp(action_xy, -1.0, 1.0)

    xy = action_xy * args_cli.drop_xy_half
    z = torch.full((num_envs, 1), args_cli.drop_z, dtype=torch.float32, device=device)

    drop_offset = torch.cat([xy, z], dim=1)
    drop_position = scene.env_origins + drop_offset

    object_state[:, cube_idx, :3] = drop_position
    object_state[:, cube_idx, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=device,
        dtype=torch.float32,
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


def check_pile_stable_for_next_drop(scene: InteractiveScene, active_cubes: int):
    """Return whether active pile is stable enough."""
    device = scene.env_origins.device

    if active_cubes <= 0:
        return (
            torch.ones(scene.num_envs, dtype=torch.bool, device=device),
            torch.zeros(scene.num_envs, dtype=torch.float32, device=device),
        )

    collection: RigidObjectCollection = scene["object_collection"]
    state_w = collection.data.object_link_state_w

    lin_vel_w = state_w[:, :active_cubes, 7:10]
    speed = torch.linalg.norm(lin_vel_w, dim=-1)

    max_speed_per_env = speed.max(dim=1).values
    pile_stable_now = max_speed_per_env <= args_cli.pile_stable_speed_thresh

    return pile_stable_now, max_speed_per_env


def update_success_mask(
    scene: InteractiveScene,
    active_cubes: int,
    stable_counter: torch.Tensor,
):
    """Check inside-bin + stable success for active cubes."""
    collection: RigidObjectCollection = scene["object_collection"]
    state_w = collection.data.object_link_state_w

    pos_w = state_w[:, :active_cubes, :3]
    lin_vel_w = state_w[:, :active_cubes, 7:10]
    pos_local = pos_w - scene.env_origins.unsqueeze(1)

    in_x = torch.abs(pos_local[..., 0]) <= args_cli.bin_half_x
    in_y = torch.abs(pos_local[..., 1]) <= args_cli.bin_half_y
    in_z = (pos_local[..., 2] >= args_cli.bin_floor_z) & (pos_local[..., 2] <= args_cli.bin_top_z)

    speed = torch.linalg.norm(lin_vel_w, dim=-1)

    stable_now = speed <= args_cli.stable_speed_thresh
    in_bin_now = in_x & in_y & in_z
    good_now = in_bin_now & stable_now

    stable_counter[:, :active_cubes] = torch.where(
        good_now,
        stable_counter[:, :active_cubes] + 1,
        torch.zeros_like(stable_counter[:, :active_cubes]),
    )

    cube_success_mask = stable_counter[:, :active_cubes] >= args_cli.stable_steps
    cubes_success_count = cube_success_mask.sum(dim=1)
    active_success_envs = torch.all(cube_success_mask, dim=1)

    return active_success_envs, stable_counter, cube_success_mask, cubes_success_count, pos_local, speed, in_bin_now


def get_vector_obs(scene: InteractiveScene, num_cubes: int, dropped_cubes: int):
    """Fixed-size vector observation."""
    collection: RigidObjectCollection = scene["object_collection"]

    state_w = collection.data.object_link_state_w
    device = state_w.device
    num_envs = scene.num_envs

    pos_w = state_w[:, :MAX_CUBES, :3]
    lin_vel_w = state_w[:, :MAX_CUBES, 7:10]
    pos_local = pos_w - scene.env_origins.unsqueeze(1)

    cube_ids = torch.arange(MAX_CUBES, device=device).view(1, MAX_CUBES)

    active_mask = (cube_ids < num_cubes).float().repeat(num_envs, 1)
    dropped_mask = (cube_ids < dropped_cubes).float().repeat(num_envs, 1)

    valid_mask = active_mask.unsqueeze(-1)
    pos_local = pos_local * valid_mask
    lin_vel_w = lin_vel_w * valid_mask

    pos_scale = torch.tensor(
        [args_cli.bin_half_x, args_cli.bin_half_y, args_cli.bin_top_z],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3)

    pos_norm = pos_local / torch.clamp(pos_scale, min=1.0e-6)
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
            active_mask,
            dropped_mask,
            progress,
        ],
        dim=1,
    )

    return obs


def get_image_obs(camera: Camera):
    """Return normalized RGB image observation."""
    rgb = camera.data.output["rgb"][..., :3]
    return rgb.float() / 255.0


def get_obs(scene: InteractiveScene, camera: Camera, num_cubes: int, dropped_cubes: int):
    """Return selected observation type."""
    if args_cli.obs_type == "vector":
        return get_vector_obs(scene, num_cubes, dropped_cubes)

    if args_cli.obs_type == "image":
        return get_image_obs(camera)

    if args_cli.obs_type == "multimodal":
        return {
            "vector": get_vector_obs(scene, num_cubes, dropped_cubes),
            "image": get_image_obs(camera),
        }

    raise ValueError(f"Unknown obs_type: {args_cli.obs_type}")


def compute_packing_metrics(pos_local: torch.Tensor, active_cubes: int):
    """Extra packing-quality metrics used for dense reward and evaluation."""
    if active_cubes <= 0:
        num_envs = pos_local.shape[0]
        device = pos_local.device
        return {
            "max_height": torch.zeros(num_envs, device=device),
            "mean_xy_dist": torch.zeros(num_envs, device=device),
            "xy_spread": torch.zeros(num_envs, device=device),
        }

    active_pos = pos_local[:, :active_cubes, :]

    xy = active_pos[..., :2]
    z = active_pos[..., 2]

    max_height = z.max(dim=1).values
    mean_xy_dist = torch.linalg.norm(xy, dim=-1).mean(dim=1)

    x_spread = active_pos[..., 0].max(dim=1).values - active_pos[..., 0].min(dim=1).values
    y_spread = active_pos[..., 1].max(dim=1).values - active_pos[..., 1].min(dim=1).values
    xy_spread = x_spread * y_spread

    return {
        "max_height": max_height,
        "mean_xy_dist": mean_xy_dist,
        "xy_spread": xy_spread,
    }


def compute_reward(
    prev_cubes_success_count: torch.Tensor,
    cubes_success_count: torch.Tensor,
    pos_local: torch.Tensor,
    speed: torch.Tensor,
    in_bin_now: torch.Tensor,
    active_success_envs: torch.Tensor,
    dropped_cubes: int,
    completion_bonus_paid: torch.Tensor | None = None,
):
    """Sparse, dense, or hybrid reward."""
    newly_successful = (cubes_success_count - prev_cubes_success_count).clamp(min=0).float()

    if dropped_cubes <= 0:
        return torch.zeros_like(newly_successful)

    active_pos = pos_local[:, :dropped_cubes, :]
    active_speed = speed[:, :dropped_cubes]

    inside_fraction = in_bin_now.float().mean(dim=1)

    metrics = compute_packing_metrics(pos_local, dropped_cubes)

    xy_ref = max(args_cli.bin_half_x, args_cli.bin_half_y, 1.0e-6)
    height_ref = max(args_cli.bin_top_z, 1.0e-6)
    area_ref = max((2.0 * args_cli.bin_half_x) * (2.0 * args_cli.bin_half_y), 1.0e-6)

    height_penalty = metrics["max_height"] / height_ref
    center_penalty = metrics["mean_xy_dist"] / xy_ref
    spread_penalty = metrics["xy_spread"] / area_ref
    speed_penalty = active_speed.mean(dim=1)

    dense_reward = (
        0.8 * inside_fraction
        - 0.20 * height_penalty
        - 0.15 * center_penalty
        - 0.10 * spread_penalty
        - 0.05 * speed_penalty
    )

    sparse_reward = newly_successful

    completion_bonus = torch.zeros_like(sparse_reward)
    if dropped_cubes == args_cli.num_cubes:
        if completion_bonus_paid is None:
            newly_completed_envs = active_success_envs
        else:
            newly_completed_envs = active_success_envs & (~completion_bonus_paid)

        completion_bonus = newly_completed_envs.float() * float(args_cli.num_cubes)
        
    if args_cli.reward_type == "sparse":
        return sparse_reward + completion_bonus

    if args_cli.reward_type == "dense":
        return dense_reward

    if args_cli.reward_type == "hybrid":
        return sparse_reward + dense_reward + completion_bonus

    raise ValueError(f"Unknown reward_type: {args_cli.reward_type}")


# RL-style environment wrapper
class BinPackingRLEnv:
    """Simple vectorized RL-style wrapper around the IsaacLab bin-packing scene.

    One environment step means:
        policy action -> drop next cube -> simulate until pile settles -> compute reward.

    Action:
        Tensor [num_envs, 2], each value in [-1, 1].

    Observation:
        Selected by --obs_type: vector, image, or multimodal.
    """

    def __init__(self, sim: SimulationContext, scene: InteractiveScene, camera: Camera):
        self.sim = sim
        self.scene = scene
        self.camera = camera
        self.collection: RigidObjectCollection = scene["object_collection"]
        self.sim_dt = sim.get_physics_dt()

        self.episode_step = 0
        self.dropped_cubes = 0

        self.stable_counter = None
        self.success_envs = None
        self.done_envs = None
        self.cube_success_mask = None
        self.cubes_success_count = None
        self.prev_cubes_success_count = None

        self.episode_reward_sum = None
        self.episode_real_t0 = None

    @property
    def num_envs(self):
        return self.scene.num_envs

    @property
    def device(self):
        return self.scene.env_origins.device

    def reset(self):
        self.stable_counter, self.success_envs, self.done_envs = reset_episode(self.sim, self.scene)

        self.episode_step = 0
        self.dropped_cubes = 0

        self.cube_success_mask = torch.zeros(
            (self.num_envs, args_cli.num_cubes),
            dtype=torch.bool,
            device=self.device,
        )
        self.cubes_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_cubes_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.completion_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_real_t0 = time.perf_counter()
        self.next_camera_save_step = args_cli.camera_interval

        self.camera.update(dt=self.sim_dt)

        return get_obs(self.scene, self.camera, args_cli.num_cubes, self.dropped_cubes)

    def _simulate_until_stable(self, active_cubes: int):
        """Simulate physics after one drop.

        For intermediate cubes:
            stop when the pile is stable enough for the next drop.

        For the final cube:
            stop only when all environments satisfy the task success condition
            or when final_settle_max_steps / max_episode_steps is reached.

        This method also updates self.stable_counter at every physics step,
        so stable_steps means consecutive physics steps, not consecutive RL steps.
        """
        pile_stable_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        final_pile_stable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        max_speed_per_env = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        last_active_success_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        last_cube_success_mask = None
        last_cubes_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        last_pos_local = None
        last_speed = None
        last_in_bin_now = None

        steps_used = 0

        max_settle_steps = (
            args_cli.final_settle_max_steps
            if active_cubes == args_cli.num_cubes
            else args_cli.settle_max_steps
        )

        for _ in range(max_settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            self.episode_step += 1
            steps_used += 1

            if args_cli.obs_type in ["image", "multimodal"] or args_cli.save_camera:
                self.camera.update(dt=self.sim_dt)

            # Pile-level stability: used to decide when it is safe to drop the next cube.
            pile_stable_now, max_speed_per_env = check_pile_stable_for_next_drop(
                self.scene,
                active_cubes=active_cubes,
            )

            pile_stable_counter = torch.where(
                pile_stable_now,
                pile_stable_counter + 1,
                torch.zeros_like(pile_stable_counter),
            )

            final_pile_stable = pile_stable_counter >= args_cli.pile_stable_steps

            # Task-level success: update every physics step.
            if active_cubes > 0 and self.stable_counter is not None:
                (
                    last_active_success_envs,
                    self.stable_counter,
                    last_cube_success_mask,
                    last_cubes_success_count,
                    last_pos_local,
                    last_speed,
                    last_in_bin_now,
                ) = update_success_mask(
                    scene=self.scene,
                    active_cubes=active_cubes,
                    stable_counter=self.stable_counter,
                )

            waited_min = steps_used >= args_cli.drop_wait_min_steps
            all_pile_stable = torch.all(final_pile_stable)

            all_task_success = (
                active_cubes == args_cli.num_cubes
                and torch.all(last_active_success_envs)
            )

            if waited_min:
                if active_cubes < args_cli.num_cubes:
                    # Before the final cube, only wait until the pile is stable
                    # enough to safely drop the next cube.
                    if all_pile_stable:
                        break
                else:
                    # After the final cube, wait for true task success.
                    if all_task_success:
                        break

            if self.episode_step >= args_cli.max_episode_steps:
                break

        settle_info = {
            "active_success_envs": last_active_success_envs,
            "cube_success_mask": last_cube_success_mask,
            "cubes_success_count": last_cubes_success_count,
            "pos_local": last_pos_local,
            "speed": last_speed,
            "in_bin_now": last_in_bin_now,
        }

        return steps_used, final_pile_stable, max_speed_per_env, settle_info
    
    def _save_camera_frame(self, force: bool = False):
        """Save camera frame.

        Uses threshold crossing instead of exact modulo because one RL step may
        advance many physics steps.
        """
        if not args_cli.save_camera:
            return

        if not force:
            if self.episode_step < self.next_camera_save_step:
                return

        os.makedirs(args_cli.camera_out_dir, exist_ok=True)

        self.camera.update(dt=self.sim_dt)
        rgb = self.camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb)
                
        if force:
            filename = f"rl_final_step{self.episode_step:06d}.png"
        else:
            filename = f"rl_step{self.episode_step:06d}.png"

        img.save(os.path.join(args_cli.camera_out_dir, filename))

        if not force:
            while self.next_camera_save_step <= self.episode_step:
                self.next_camera_save_step += args_cli.camera_interval

    def step(self, action_xy):
        """Drop next cube using action and return obs, reward, done, info."""
        if self.done_envs is not None and torch.all(self.done_envs):
            obs = get_obs(self.scene, self.camera, args_cli.num_cubes, self.dropped_cubes)
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            info = {
                "dropped_cubes": self.dropped_cubes,
                "episode_step": self.episode_step,
                "settle_steps": 0,
                "success_envs": self.success_envs.detach().clone(),
                "timeout_envs": torch.zeros_like(self.success_envs),
                "episode_reward_sum": self.episode_reward_sum.detach().clone(),
                "already_done": True,
            }
            return obs, reward, self.done_envs.detach().clone(), info
        
        if isinstance(action_xy, np.ndarray):
            action_xy = torch.tensor(action_xy, dtype=torch.float32, device=self.device)
        elif not isinstance(action_xy, torch.Tensor):
            action_xy = torch.tensor(action_xy, dtype=torch.float32, device=self.device)

        action_xy = action_xy.to(device=self.device, dtype=torch.float32)

        if self.dropped_cubes < args_cli.num_cubes:
            drop_cube_at_action(self.scene, self.dropped_cubes, action_xy)
            self.dropped_cubes += 1

        settle_steps, final_stable, max_speed_per_env, settle_info = self._simulate_until_stable(
            active_cubes=self.dropped_cubes,
        )

        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        active_success_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        info = {
            "dropped_cubes": self.dropped_cubes,
            "episode_step": self.episode_step,
            "settle_steps": settle_steps,
            "pile_stable": final_stable.detach().clone(),
            "max_speed_per_env": max_speed_per_env.detach().clone(),
        }

        if self.dropped_cubes > 0:
            active_success_envs = settle_info["active_success_envs"]
            active_cube_success_mask = settle_info["cube_success_mask"]
            active_cubes_success_count = settle_info["cubes_success_count"]
            active_pos_local = settle_info["pos_local"]
            active_speed = settle_info["speed"]
            in_bin_now = settle_info["in_bin_now"]

            # Fallback in case no physics step was taken.
            if active_cube_success_mask is None:
                (
                    active_success_envs,
                    self.stable_counter,
                    active_cube_success_mask,
                    active_cubes_success_count,
                    active_pos_local,
                    active_speed,
                    in_bin_now,
                ) = update_success_mask(
                    scene=self.scene,
                    active_cubes=self.dropped_cubes,
                    stable_counter=self.stable_counter,
                )

            self.cube_success_mask[:, :self.dropped_cubes] = active_cube_success_mask
            self.cubes_success_count = active_cubes_success_count

            reward = compute_reward(
                prev_cubes_success_count=self.prev_cubes_success_count,
                cubes_success_count=active_cubes_success_count,
                pos_local=active_pos_local,
                speed=active_speed,
                in_bin_now=in_bin_now,
                active_success_envs=active_success_envs,
                dropped_cubes=self.dropped_cubes,
                completion_bonus_paid=self.completion_bonus_paid,
            )

            if self.dropped_cubes == args_cli.num_cubes:
                self.completion_bonus_paid |= active_success_envs

            self.prev_cubes_success_count = active_cubes_success_count.clone()
            self.episode_reward_sum += reward

            metrics = compute_packing_metrics(active_pos_local, self.dropped_cubes)
            info["cubes_success_count"] = active_cubes_success_count.detach().clone()
            info["cube_success_mask"] = active_cube_success_mask.detach().clone()
            info["max_height"] = metrics["max_height"].detach().clone()
            info["mean_xy_dist"] = metrics["mean_xy_dist"].detach().clone()
            info["xy_spread"] = metrics["xy_spread"].detach().clone()
        
        if self.dropped_cubes == args_cli.num_cubes:
            self.success_envs = active_success_envs

        timed_out = self.episode_step >= args_cli.max_episode_steps
        timeout_envs = (~self.success_envs) & torch.full_like(self.success_envs, timed_out)
        self.done_envs = self.success_envs | timeout_envs

        info["success_envs"] = self.success_envs.detach().clone()
        info["timeout_envs"] = timeout_envs.detach().clone()
        info["episode_reward_sum"] = self.episode_reward_sum.detach().clone()

        self._save_camera_frame()

        obs = get_obs(self.scene, self.camera, args_cli.num_cubes, self.dropped_cubes)

        return obs, reward, self.done_envs.detach().clone(), info

    def random_action(self):
        return torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0

    def close(self):
        pass


# Main random-policy 
def run_random_policy_test(env: BinPackingRLEnv):
    print("[INFO] Running BinPackingRLEnv random-policy validation", flush=True)
    print(f"[INFO] num_envs = {args_cli.num_envs}", flush=True)
    print(f"[INFO] num_cubes = {args_cli.num_cubes}", flush=True)
    print(f"[INFO] cube_size = {args_cli.cube_size}", flush=True)
    print(f"[INFO] obs_type = {args_cli.obs_type}", flush=True)
    print(f"[INFO] reward_type = {args_cli.reward_type}", flush=True)
    print(f"[INFO] action = normalized [x, y] in [-1, 1]", flush=True)

    for ep in range(args_cli.num_episodes):
        print(f"\n{'=' * 80}", flush=True)
        print(f"[INFO] Episode {ep + 1}/{args_cli.num_episodes}", flush=True)

        obs = env.reset()

        if isinstance(obs, dict):
            print(f"[OBS ] vector shape = {obs['vector'].shape}", flush=True)
            print(f"[OBS ] image shape  = {obs['image'].shape}", flush=True)
        else:
            print(f"[OBS ] {args_cli.obs_type} shape = {obs.shape}", flush=True)

        ep_t0 = time.perf_counter()

        while simulation_app.is_running():
            action = env.random_action()
            obs, reward, done, info = env.step(action)

            cubes_success_count = info.get("cubes_success_count", None)
            if cubes_success_count is not None:
                mean_cubes_success = cubes_success_count.float().mean().item()
                min_cubes_success = int(cubes_success_count.min().item())
                max_cubes_success = int(cubes_success_count.max().item())
                cube_success_text = (
                    f"cube_success_mean={mean_cubes_success:.1f}/{args_cli.num_cubes} | "
                    f"cube_success_minmax={min_cubes_success}-{max_cubes_success}"
                )
            else:
                cube_success_text = "cube_success_mean=N/A"

            print(
                f"[STEP] dropped={info['dropped_cubes']:02d}/{args_cli.num_cubes} | "
                f"sim_step={info['episode_step']:04d} | "
                f"settle_steps={info['settle_steps']:03d} | "
                f"reward_mean={reward.mean().item(): .4f} | "
                f"{cube_success_text} | "
                f"episode_success={int(info['success_envs'].sum().item())}/{env.num_envs} | "
                f"done={int(done.sum().item())}/{env.num_envs}",
                flush=True,
            )

            if torch.all(done):
                break

        ep_wall = time.perf_counter() - ep_t0
        sim_time = env.episode_step * env.sim_dt
        fps = env.episode_step / max(ep_wall, 1.0e-6)
        rtf = sim_time / max(ep_wall, 1.0e-6)

        env._save_camera_frame(force=True)

        print("[INFO] Episode finished", flush=True)
        print(f"[INFO] sim steps: {env.episode_step}", flush=True)
        print(f"[INFO] sim time:  {sim_time:.4f} s", flush=True)
        print(f"[INFO] wall time: {ep_wall:.4f} s", flush=True)
        print(f"[INFO] FPS:       {fps:.2f}", flush=True)
        print(f"[INFO] RTF:       {rtf:.3f}", flush=True)
        print(f"[INFO] reward sum per env:\n{env.episode_reward_sum}", flush=True)
        print(f"[INFO] success envs: {int(env.success_envs.sum().item())}/{env.num_envs}", flush=True)
        print(f"[INFO] cubes succeeded per env:\n{env.cubes_success_count}", flush=True)


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

    camera = make_debug_camera(sim)

    sim.reset()

    camera.reset()
    camera.update(dt=sim.get_physics_dt())

    cam_pos = torch.tensor([[0.0, -2.6, 4.8]], device=sim.device)
    cam_target = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(cam_pos, cam_target)
    camera.update(dt=sim.get_physics_dt())

    sim.play()

    env = BinPackingRLEnv(sim=sim, scene=scene, camera=camera)

    run_random_policy_test(env)

    simulation_app.close()


if __name__ == "__main__":
    main()