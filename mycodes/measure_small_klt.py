import argparse

from omni.isaac.lab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Measure small_KLT bin dimensions")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after app launch
from pxr import Usd, UsdGeom
import omni.usd

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext, schemas
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


# Scene config
class MeasureBinSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    bin = AssetBaseCfg(
        prim_path="/World/envs/env_0/small_KLT",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.0, 2.0, 2.0),   # same scale as your current code
            rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(num_envs=1, env_spacing=2.0, **kwargs)


# Helper to print bbox
def print_prim_bbox(prim_path: str, applied_scale=(2.0, 2.0, 2.0)):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"[ERROR] Prim not found: {prim_path}")
        return

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedBox()

    min_pt = aligned_box.GetMin()
    max_pt = aligned_box.GetMax()
    size = max_pt - min_pt

    print("\n" + "=" * 80)
    print(f"[INFO] Prim path: {prim_path}")
    print(f"[INFO] World bbox min: {min_pt}")
    print(f"[INFO] World bbox max: {max_pt}")
    print(f"[INFO] World bbox size (scaled): {size}")

    # estimate original size from known uniform scale
    sx, sy, sz = applied_scale
    est_original = (size[0] / sx, size[1] / sy, size[2] / sz)

    print(f"[INFO] Applied scale: {applied_scale}")
    print(f"[INFO] Estimated original asset size: {est_original}")
    print("=" * 80 + "\n")



# Main
def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    scene_cfg = MeasureBinSceneCfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Scene loaded. Measuring bin bounding box...")
    print_prim_bbox("/World/envs/env_0/small_KLT", applied_scale=(2.0, 2.0, 2.0))

    # keep app alive for one frame so logs flush nicely
    sim.step()

    simulation_app.close()


if __name__ == "__main__":
    main()