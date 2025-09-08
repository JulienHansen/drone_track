import os
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

current_dir = os.getcwd()
CUSTOM_GATE_USD_PATH = os.path.join(current_dir, "assets", "gate", "gate.usd")

def generate_track(track_config: dict | None) -> RigidObjectCollectionCfg:
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{gate_id}": RigidObjectCfg(
                prim_path=f"/World/Gate_{gate_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path = CUSTOM_GATE_USD_PATH,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(gate_config["pos"][0], gate_config["pos"][1], gate_config["pos"][2] - 1),
                    rot=math_utils.quat_from_euler_xyz(
                        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(gate_config["yaw"])
                    ).tolist(),
                ),
            )
            for gate_id, gate_config in track_config.items()
        }
    )

