import argparse
import time
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Cube Drop Simulation")
#parser.add_argument("--device", type=str, default="cuda", help="Device to run simulation on")
parser.add_argument("--num_cubes", type=int, default=20, help="Number of cubes to drop")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Environment spacing")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
import torch
import omni.isaac.core.utils.prims as prim_utils
import numpy as np
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg,RigidObjectCollectionCfg,RigidObjectCollection
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim import converters, schemas
from typing import List
from omni.isaac.lab.utils import configclass

class CubeDropSceneCfg(InteractiveSceneCfg):
    """Spawns 20 cubes under /World/Origin{i} in each of num_envs environments."""

    # ground + light
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5))
    )

    bin: AssetBaseCfg = AssetBaseCfg(
    prim_path="/World/envs/env_.*/small_KLT",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
        scale=(2, 2, 2),
        rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
        fixed_tendons_props=True,
        activate_contact_sensors=True,
    )
    )


    # build scene_entities directly as a class attribute
    
    object_collection:RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    rigid_objects={
        # build one entry per cube
        **{
            f"cube_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Origin{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        metallic=0.2
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 2.0)  # or whatever per‐cube init you like
                ),
            )
            for i in range(20)
        }
    }
)

    def __init__(self,
                 *,
                 num_envs: int = 2,
                 env_spacing: float = 2.0,
                 **kwargs):
        # only pass what the base expects
        super().__init__(num_envs=num_envs,
                         env_spacing=env_spacing,
                         **kwargs)
    

    def _build(self, simulator: SimulationContext):
        # let the parent spawn ground + light + all scene_entities
        for env_idx in range(self.num_envs):
            for i in range(20):
                prim_utils.create_prim(
                    f"/World/envs/env_{env_idx}/Origin{i}",
                    "Xform",
                    translation=[0, 0, 0],
                )
        super()._build(simulator)

        # now put your origin on the device so run_simulator can use it
        self.env_origins = torch.tensor(self._origins,
                                        device=simulator.device)
        
def run_simulator(sim: SimulationContext,
                  scene: InteractiveScene,
                  batch_size: int = 20,
                  num_batches: int = 10):
    """Drop `batch_size` cubes per batch, repeat for `num_batches` batches,
    across all environments in `scene`.

    For each cube, print:
    - simulation time advanced before the next cube
    - real wall-clock time spent on this machine
    - cumulative simulation time
    """
    collection: RigidObjectCollection = scene["object_collection"]
    sim_dt = sim.get_physics_dt()
    final_positions = []

    # total simulated time since run_simulator started
    total_sim_time = 0.0

    # ensure physics is running
    sim.play()

    print(f"[INFO]: Physics dt = {sim_dt:.6f} s")

    for batch in range(1, num_batches + 1):
        print(f"\n[INFO]: Starting Batch {batch}")

        for i in range(batch_size):
            ids, names = collection.find_objects(f"cube_{i}", preserve_order=True)
            obj_id = ids[0].item()

            # clone the default root state ([num_envs, num_objects, state_dim])
            object_state = collection.data.default_object_state.clone()

            # sample per-env xy in [-0.2, +0.2], z in [0.25, 0.5]
            num_envs = scene.num_envs
            dev_str = str(object_state.device)

            xy = math_utils.sample_uniform(
                -0.2, 0.2,
                size=(num_envs, 2),
                device=dev_str
            )
            z = math_utils.sample_uniform(
                0.25, 0.5,
                size=(num_envs, 1),
                device=dev_str
            )

            drop_offset = torch.cat([xy, z], dim=1)         # [num_envs, 3]
            drop_position = scene.env_origins + drop_offset # [num_envs, 3]

            # write pose
            object_state[:, obj_id, :3] = drop_position
            quat = torch.tensor([0, 0, 0, 1], device=object_state.device).view(1, 4).repeat(num_envs, 1)
            object_state[:, obj_id, 3:7] = quat

            single_pose = object_state[:, obj_id, :7].unsqueeze(1)
            collection.write_object_link_pose_to_sim(single_pose, object_ids=ids)

            # timing starts here
            # -----------------------------
            real_t0 = time.perf_counter()
            sim_time_this_cube = 0.0

            # step physics for 1 simulated second
            num_steps = int(1.0 / sim_dt)
            for _ in range(num_steps):
                sim.step()
                sim_time_this_cube += sim_dt

            real_t1 = time.perf_counter()
            total_sim_time += sim_time_this_cube
            
            # timing ends here
            # -----------------------------

            # record landings for each env
            landed = collection.data.object_link_state_w[:, obj_id, :3].cpu().numpy()
            final_positions.append(landed)

            print(f"[RESULT]: cube_{i} landed at {landed}")
            print(
                f"[TIME ]: cube_{i} | "
                f"sim_this_drop = {sim_time_this_cube:.4f} s | "
                f"real_this_drop = {real_t1 - real_t0:.4f} s | "
                f"total_sim_time = {total_sim_time:.4f} s"
            )

        # reset all envs for the next batch
        print(f"\n[INFO]: Resetting simulation state after Batch {batch}...")
        collection.reset()
        sim.reset(soft=True)

    return final_positions



def main():
    parser = argparse.ArgumentParser(description="Cube Drop Simulation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run simulation on")

    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
    parser.add_argument("--env_spacing", type=float, default=2.0, help="Environment spacing")
    
    args_cli = parser.parse_args()
    # 1) Initialize simulator
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim    = SimulationContext(sim_cfg)
    sim.set_camera_view( eye   =[0.0, -3.0, 5.0], target=[0.0,  0.0, 0.75])  # type: ignore
    # 2) Build your interactive scene with N environments
    scene_cfg = CubeDropSceneCfg(
        num_envs    = args_cli.num_envs,    # what the base class wants
        env_spacing = args_cli.env_spacing, # what the base class wants
    )
    scene = InteractiveScene(scene_cfg)

    # 3) Reset and start
    sim.reset()
    print("[INFO]: Setup complete…")

    # 4) Run your batches
    run_simulator(
        sim,
        scene,
    )

    # 5) Optionally do something with `positions`…
    print("All batches complete. Collected positions for analysis.")

    # 6) Clean up
    simulation_app.close()


if __name__ == "__main__":
    main()
