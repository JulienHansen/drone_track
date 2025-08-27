# ./isaaclab.sh -p scripts/my_gate_demo.py
from isaaclab.app import AppLauncher
app = AppLauncher().app  # launch Isaac

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

def spawn_racing_gate(
    prim_path: str,
    center=(0.0, 0.0, 1.5),        # x,y,z
    inner_size=(1.5, 1.2),         # width (y), height (z) of the OPENING, in meters
    bar_thickness=0.05,            # thickness along Y for horizontals / along Y for verticals
    depth=0.05,                    # thickness along X (gate “depth”)
    color=(0.0, 0.5, 1.0),         # RGB 0..1
    kinematic=True,                # True -> fixed in place
    collision=True,                # True -> drone can hit the frame
):
    # parent Xform
    prim_utils.create_prim(prim_path, "Xform")

    w, h = inner_size
    cx, cy, cz = center

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

    # We orient the gate so: X = depth, Y = width, Z = height.
    # Vertical side bars (tall along Z): size=(depth, bar_thickness, height)
    bar("left",  (depth, bar_thickness, h), (cx, cy - half_w, cz))
    bar("right", (depth, bar_thickness, h), (cx, cy + half_w, cz))

    # Horizontal bars (wide along Y): size=(depth, width, bar_thickness)
    bar("top",    (depth, w, bar_thickness), (cx, cy, cz + half_h))
    bar("bottom", (depth, w, bar_thickness), (cx, cy, cz - half_h))

def main():
    # basic sim setup
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view([3.0, 0.0, 2.0], [0.0, 0.0, 1.2])

    # ground + lighting (optional niceties)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/sun", sim_utils.DistantLightCfg(), translation=(1,0,10))

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
    )

    sim.reset()
    print("[INFO] Gate spawned. Fly through the opening!")

if __name__ == "__main__":
    main()
