# ./isaaclab.sh -p scripts/my_gate_demo.py
from isaaclab.app import AppLauncher
app = AppLauncher().app  # launch Isaac

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
# Required for Euler to Quaternion conversion
from scipy.spatial.transform import Rotation

def spawn_racing_gate(
    prim_path: str,
    center=(0.0, 0.0, 1.5),
    inner_size=(1.5, 1.2),
    bar_thickness=0.05,
    depth=0.05,
    color=(0.0, 0.5, 1.0),
    kinematic=True,
    collision=True,
    # This now correctly accepts Euler angles in radians (roll, pitch, yaw)
    rotation_euler_xyz=(0.0, 0.0, 0.0),
):
    # 1. Convert Euler angles to a quaternion
    # Scipy creates a quaternion in (x, y, z, w) format
    rot_quat_xyzw = Rotation.from_euler('xyz', rotation_euler_xyz, degrees=False).as_quat()
    # Isaac Sim expects quaternion in (w, x, y, z) format, so we reorder it
    rot_quat_wxyz = (rot_quat_xyzw[3], rot_quat_xyzw[0], rot_quat_xyzw[1], rot_quat_xyzw[2])

    # 2. Create the parent Xform with the correct 'orientation' parameter
    prim_utils.create_prim(prim_path, "Xform", translation=center, orientation=rot_quat_wxyz)

    w, h = inner_size
    
    # materials / physics
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic) if collision else None
    coll  = sim_utils.CollisionPropertiesCfg() if collision else None

    def bar(name, size_xyz, pos_xyz):
        cfg = sim_utils.CuboidCfg(
            size=size_xyz,
            visual_material=mat,
            rigid_props=rigid,
            collision_props=coll,
        )
        cfg.func(f"{prim_path}/{name}", cfg, translation=pos_xyz)

    half_w = w * 0.5
    half_h = h * 0.5

    # The rest of the function is the same...
    # vertical bars
    bar("left",  (depth, bar_thickness, h), (0.0, -half_w, 0.0))
    bar("right", (depth, bar_thickness, h), (0.0, half_w, 0.0))

    # horizontal bars
    bar("top",    (depth, w, bar_thickness), (0.0, 0.0, half_h))
    bar("bottom", (depth, w, bar_thickness), (0.0, 0.0, -half_h))

def main():
    # basic sim setup
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view([3.0, 0.0, 2.0], [0.0, 0.0, 1.2])

    # ground + lighting
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/sun", sim_utils.DistantLightCfg(), translation=(1, 0, 10))

    # spawn one gate
    spawn_racing_gate(
        "/World/Gate1",
        center=(0.0, 0.0, 1.2),
        inner_size=(1.6, 1.2),
        bar_thickness=0.06,
        depth=0.06,
        color=(0.1, 0.7, 1.0),
        kinematic=True,
        collision=True,
        rotation_euler_xyz=(0.5, 0.3, 0.0)
    )

    sim.reset()
    print("[INFO] Gate spawned with rotation applied!")

    # --- Keep simulation alive ---
    while app.is_running():
        sim.step()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
