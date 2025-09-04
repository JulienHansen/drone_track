# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom drone (URDF-based)."""

from __future__ import annotations
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
current_dir = os.getcwd()
CUSTOM_DRONE_URDF_PATH = os.path.join(current_dir, "assets", "Custom_drone", "Custom_drone.urdf")

# ------------------------------------------------------------------
# URDF import configuration
# ------------------------------------------------------------------
import_config = _urdf.ImportConfig()
import_config.fix_base = False           # Drone non fixé au sol
import_config.make_default_prim = False  # Le prim path sera géré via ArticulationCfg
import_config.self_collision = False
import_config.convex_decomp = False
import_config.density = 0.0

# ------------------------------------------------------------------
# Parse URDF
# ------------------------------------------------------------------
urdf_interface = _urdf.acquire_urdf_interface()

result, robot_model = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path=CUSTOM_DRONE_URDF_PATH,
    import_config=import_config
)
if not result:
    raise RuntimeError(f"[ERROR] Unable to parse the URDF: {CUSTOM_DRONE_URDF_PATH}")

# ------------------------------------------------------------------
# Import robot (model only)
# ------------------------------------------------------------------
result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_robot=robot_model,
    import_config=import_config
)
if not result:
    raise RuntimeError(f"[ERROR] Unable to import the drone URDF into the scene.")

print(f"[INFO] Custom drone initially imported at prim path: {prim_path}")

# ------------------------------------------------------------------
# Déplacer le prim vers un path multi-env compatible
# ------------------------------------------------------------------
new_prim_path = "/World/envs/env_0/Robot"
omni.kit.commands.execute("MovePrim", path_from=prim_path, path_to=new_prim_path)
print(f"[INFO] Custom drone moved to: {new_prim_path}")

# ------------------------------------------------------------------
# Configuration finale pour Isaac Lab
# ------------------------------------------------------------------
CUSTOM_DRONE_CFG = ArticulationCfg(
    prim_path=new_prim_path,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={
            "propeller_1_joint": 0.0,
            "propeller_2_joint": -0.0,
            "propeller_3_joint": -0.0,
            "propeller_4_joint": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

print("[INFO] CUSTOM_DRONE_CFG ready for use.")

