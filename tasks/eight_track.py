from __future__ import annotations


import torch
from torch.func import vmap
import torch.distributions as D
import torch.nn.functional as F
import gymnasium as gym
from utils.math import *
from assets.track_generator import * 
from collections import deque
import yaml
import os 


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms


#from assets.crazyflie import CRAZYFLIE_CFG
#from assets.five_in_drone import FIVE_IN_DRONE 
from assets.Custom_drone import CUSTOM_DRONE_CFG

from isaaclab.markers import CUBOID_MARKER_CFG
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.core.utils.viewports import set_camera_view



@configclass
class EightEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 1
    action_space = 4
    observation_space = 22
    state_space = 0
    moment_scale = 2
    debug_vis = True 
    reset_tracking_error_threshold = .5  
    reset_height_threshold = 0.15  


    sim: SimulationCfg = SimulationCfg(
        dt= 1/100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2000, env_spacing=15.0, replicate_physics=True)

    robot: ArticulationCfg = CUSTOM_DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 3.5

class EightEnv(DirectRLEnv):
    cfg: EightEnvCfg

    def __init__(self, cfg: EightEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.future_traj_steps = 4
        self.traj_t0 = torch.zeros(self.num_envs, device=self.device)
        self.origin = torch.tensor([0., 0., 2.], device=self.device)
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_signed = torch.zeros(self.num_envs, device=self.device)
        self.prev_dist_gate = torch.zeros(self.num_envs, device=self.device)


        self._body_id = self._robot.find_bodies("drone_body")[0] # body for crazyflie, drone_body for the others 
        self.rpos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in ["reward_pose", "reward_effort" ,"reward_up"]
        }
        self.set_debug_vis(self.cfg.debug_vis)
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _setup_scene(self):
        # --- robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        current_dir = os.getcwd()
        GATE_CONFIG_PATH = os.path.join(current_dir, "assets", "gate", "gates_config.yaml")

        # Lecture du YAML
        with open(GATE_CONFIG_PATH, "r") as f:
            gates_data = yaml.safe_load(f)

        gate_config = {}

        for gate in sorted(gates_data.get("gates", []), key=lambda g: g["id"]):
            gate_id = str(gate["id"])
            gate_config[gate_id] = {
                "pos": gate["position"],       # [x, y, z]
                "rot": gate["orientation"]     # [roll, pitch, yaw]
            }
        
        # Génération du track à partir de la liste chaînée
        track_cfg: RigidObjectCollectionCfg = generate_track(track_config=gate_config)    

        self.track: RigidObjectCollection = track_cfg.class_type(track_cfg)


        # Récupérer toutes les positions [x, y, z]
        self.gate_pos = torch.tensor(
            [gate_config["pos"] for _, gate_config in gate_config.items()],
            device=self.device
        )  # [G, 3]

        # Récupérer toutes les rotations [roll, pitch, yaw]
        self.gate_rot = torch.tensor(
            [gate_config["rot"] for _, gate_config in gate_config.items()],
            device=self.device
        )  # [G, 3]

        # Créer les quaternions à partir des rotations Euler (roll, pitch, yaw)
        quats = math_utils.quat_from_euler_xyz(
            self.gate_rot[:, 0],  # roll
            self.gate_rot[:, 1],  # pitch
            self.gate_rot[:, 2],  # yaw
        )  # [G, 4]

        # Vecteur normal local (par exemple l'axe X de la gate)
        local_normal = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(quats), 1)  # [G, 3]

        # Appliquer la rotation à ce vecteur
        self.gate_normal = math_utils.quat_apply(quats, local_normal)  # [G, 3]
        self.gate_radius = 0.75  # half of inner size ~1.5

    

        starting_positions = gates_data["starting_positions"]  # liste de listes [[x,y,z], ...]

        # Conversion en tenseur sur le bon device
        self.start_pos = torch.tensor(starting_positions, device=self.device)  # [N, 3]

        # --- terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # --- clone environments
        self.scene.clone_environments(copy_from_source=False)

        # --- global light (Option B)
        # --- global dome light
        light_cfg = sim_utils.DomeLightCfg(
            intensity=1000.0,      # stronger intensity
            color=(0.75, 0.75, 0.75),     # neutral grey
        )
        light_cfg.func("/World/global_light", light_cfg) 



    # Currently working on the new physic model i will try to not destroyed abything cut i cannot promise 
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        drone_pos  = self._robot.data.root_link_pos_w
        drone_quat = self._robot.data.root_link_quat_w
        lin_vel    = self._robot.data.root_lin_vel_b
        ang_vel    = self._robot.data.root_ang_vel_b

        # replace quaternion_to_euler with whatever util you use; keep 3-dim Euler
        drone_euler = quaternion_to_euler(drone_quat)       # [N, 3]
        rotor_cmds  = (self._actions + 1.0) / 2.0

        gid  = self.progress_buf                           # [N]
        gidn = (gid + 1) % self.gate_pos.shape[0]

        curr_gate_pos = self.gate_pos[gid]                 # [N, 3]
        next_gate_pos = self.gate_pos[gidn]                # [N, 3]

        # gate-in-drone frame positions
        curr_gate_rel_b, _ = subtract_frame_transforms(drone_pos, drone_quat, curr_gate_pos, torch.zeros_like(drone_quat))
        next_gate_rel_b, _ = subtract_frame_transforms(drone_pos, drone_quat, next_gate_pos, torch.zeros_like(drone_quat))

        next_gate_rot = self.gate_rot[gidn]       # [N, 3] -> roll, pitch, yaw

        obs = torch.cat([curr_gate_rel_b, lin_vel, drone_euler, ang_vel, rotor_cmds, next_gate_rel_b, next_gate_rot], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        device = self.device
        N = self.num_envs

        p = self._robot.data.root_pos_w                     # [N, 3]
        if not hasattr(self, "prev_pos"):
            self.prev_pos = p.clone()

        gid  = self.progress_buf                            # [N]
        gpos = self.gate_pos[gid]                           # [N, 3]
        gN   = self.gate_normal[gid]                        # [N, 3]

        # --- signed distance to gate plane & radial distance to center
        diff = p - gpos                                     # [N, 3]
        signed = torch.sum(diff * gN, dim=1)                # [N]
        radial = torch.norm(diff, dim=1)                    # [N]

        # --- detect crossing (inside aperture + plane crossed forward)
        inside = radial <= self.gate_radius
        crossed = (self.prev_signed <= 0.0) & (signed > 0.0) & inside

        # increment gate (wrap automatically)
        if crossed.any():
            self.progress_buf[crossed] = (self.progress_buf[crossed] + 1) % self.gate_pos.shape[0]
            # update references for those envs (optional: reset prev_dist to avoid spike)
            new_gid = self.progress_buf[crossed]
            self.prev_dist_gate[crossed] = torch.norm(p[crossed] - self.gate_pos[new_gid], dim=1)

        # store for next step
        self.prev_signed = signed.detach()

        # ----------------------------------------------------------------
        # Rewards (shaped like the paper, + some light shaping for stability)
        # ----------------------------------------------------------------
        # 1) progress toward current gate center
        curr_dist = torch.norm(p - self.gate_pos[self.progress_buf], dim=1)        # [N]
        prev_dist = self.prev_dist_gate if hasattr(self, "prev_dist_gate") else curr_dist.detach()
        progress_reward = (prev_dist - curr_dist).unsqueeze(1)                     # [N, 1]

        # 2) small bonus on actual crossing (helps commit through the gate)
        pass_bonus = (crossed.float() * 1.0).unsqueeze(1)                           # +1.0 when passed

        # 3) angular-rate penalty (as in paper)
        omega = self._robot.data.root_ang_vel_b
        rate_penalty = 0.001 * torch.norm(omega, dim=1, keepdim=True)

        # 4) tiny step penalty to keep moving
        step_penalty = 0.0005 * torch.ones((N, 1), device=device)

        reward = progress_reward + pass_bonus - rate_penalty - step_penalty

        # Safety / OOB
        out_of_bounds = (
            (p[:, 0] < -5) | (p[:, 0] > 5) |
            (p[:, 1] < -5) | (p[:, 1] > 5) |
            (p[:, 2] <  0) | (p[:, 2] > 7)
        )
        collided = p[:, 2] < 0.2
        crashed = out_of_bounds | collided
        reward[crashed] = -10.0

        # book-keeping for next step
        self.prev_pos = p.clone()
        self.prev_dist_gate = curr_dist.detach()

        return reward
  
           
    def _get_dones(self) -> tuple[torch.tensor, torch.tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            height_too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.reset_height_threshold
            height_too_high = self._robot.data.root_pos_w[:, 2] > 200.0 

            died = torch.logical_or(height_too_low, height_too_high)
        

            return died, time_out

    def _reset_idx(self, env_ids: torch.tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums.keys():
            avg_sum = torch.mean(self._episode_sums[key][env_ids])
            extras[f"episode_reward/{key}"] = avg_sum / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras["episode_termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["episode_termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        root_pose = torch.zeros((len(env_ids), 7), device=self.device)

        num_starts = len(self.start_pos)

        # choose spawn and set initial current gate to the closest one
        for i, env_id in enumerate(env_ids):
            start_id = torch.randint(0, num_starts, (1,), device=self.device).item()
            sx, sy, sz = self.start_pos[start_id]

            # nearest gate id w.r.t. (x,y)
            diffs = self.gate_pos - torch.tensor([sx, sy, sz], device=self.device)  # [G, 3]
            closest_gate_id = torch.argmin(torch.sum(diffs * diffs, dim=1)).item()

            roll, pitch, yaw = self.gate_rot[closest_gate_id]  # [roll, pitch, yaw]

            roll = roll.to(self.device) if isinstance(roll, torch.Tensor) else torch.tensor(roll, device=self.device)
            pitch = pitch.to(self.device) if isinstance(pitch, torch.Tensor) else torch.tensor(pitch, device=self.device)
            yaw = yaw.to(self.device) if isinstance(yaw, torch.Tensor) else torch.tensor(yaw, device=self.device)

            spawn_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)


            root_pose[i, :3] = torch.tensor([sx, sy, sz], device=self.device)
            root_pose[i, 3:] = spawn_quat

            # set current gate and init signed distance wrt that gate
            self.progress_buf[env_id] = closest_gate_id
            gate_pos_i = self.gate_pos[closest_gate_id]
            gate_norm_i = self.gate_normal[closest_gate_id]
            self.prev_signed[env_id] = torch.dot(torch.tensor([sx, sy, sz], device=self.device) - gate_pos_i, gate_norm_i)
            self.prev_dist_gate[env_id] = torch.norm(torch.tensor([sx, sy, sz], device=self.device) - gate_pos_i)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids) 
