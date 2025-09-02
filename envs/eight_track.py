from __future__ import annotations


import torch
from torch.func import vmap
import torch.distributions as D
import torch.nn.functional as F
import gymnasium as gym
from utils import *

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms


from drone_assets.crazyflie import CRAZYFLIE_CFG

#from drone_assets.Custom_drone import CUSTOM_DRONE_CFG

from isaaclab.markers import CUBOID_MARKER_CFG
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.core.utils.viewports import set_camera_view

gate_features = [
    ((0.0, -3.0, 4.0),(0.0, 0.0, 0.0)),
    ((-1.5, -1.5, 4.0),(0.0, 0.0, torch.pi/2)),
    ((0.0, 0.0, 4.0),(0.0, 0.0, 0.0)),
    ((1.5, 1.5, 4.0),(0.0, 0.0, torch.pi/2)),
    ((0.0, 3.0, 4.0),(0.0, 0.0, 0.0)),
    ((-1.5, 1.5, 4.0),(0.0, 0.0, torch.pi/2)),
    ((0.0, 0.0, 4.0),(0.0, 0.0, 0.0)),
    ((1.5, -1.5, 4.0),(0.0, 0.0, torch.pi/2)),
]

starting_position = [
    (1.0, -3.0, 4.0),
    (-1.5, -2.5 , 4.0),
    (-1.0, 0.0, 4.0),
    (1.5, 0.5, 4.0),
    (1.0, 3.0, 4.0),
    (-1.5, 2.5, 4.0),
    (1.5, -0.5, 4.0),
]



@configclass
class EightEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 1
    action_space = 4
    observation_space = 20
    state_space = 0
    moment_scale = 0.02
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

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9

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
        self._body_id = self._robot.find_bodies("drone_body")[0]
        self.rpos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in ["reward_pose", "reward_effort" ,"reward_up"]
        }
        self.set_debug_vis(self.cfg.debug_vis)
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        for idx, (position, angle) in enumerate(gate_features, start=1):
            gate_path = f"/World/Gate{idx}"
            spawn_racing_gate(
                gate_path,
                center=position,
                inner_size=(1.5, 1.5),
                bar_thickness=0.06,
                depth=0.06,
                color=(0.1, 0.7, 1.0),
                kinematic=True,
                collision=True,
                rotation_euler_xyz=angle
    )

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # Currently working on the new physic model i will try to not destroyed abything cut i cannot promise 
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # Get drone state
        drone_pos = self._robot.data.root_link_pos_w        # [num_envs, 3]
        drone_quat = self._robot.data.root_link_quat_w       # [num_envs, 4]
        lin_vel = self._robot.data.root_lin_vel_b      # [num_envs, 3]
        ang_vel = self._robot.data.root_ang_vel_b      # [num_envs, 3]
        drone_euler = quaternion_to_euler(drone_quat)
        rotor_cmds = (self._actions + 1.0) / 2.0      # normalize [-1, 1] -> [0, 1]

        # Build tensor of gate positions
        gate_positions = torch.tensor([g[0] for g in gate_features], device=self.device)
        diff = gate_positions.unsqueeze(0) - drone_pos.unsqueeze(1)
        dists = torch.norm(diff, dim=2)

        # Find current and next gate indices
        curr_gate_ids = torch.argmin(dists, dim=1)
        next_gate_ids = (curr_gate_ids + 1) % len(gate_features)

        # Current & next gate world positions
        curr_gate_pos = gate_positions[curr_gate_ids]
        next_gate_pos = gate_positions[next_gate_ids]

         # Use Isaac Lab utility to compute relative positions in drone frame
        curr_gate_rel_b, _ = subtract_frame_transforms(
            drone_pos, drone_quat,
            curr_gate_pos, torch.zeros_like(drone_quat)
        )
        next_gate_rel_b, _ = subtract_frame_transforms(
            drone_pos, drone_quat,
            next_gate_pos, torch.zeros_like(drone_quat)
        )

        # Get gate orientations (yaw only)
        gate_orientations = torch.tensor([g[1] for g in gate_features], device=self.device)
        next_gate_yaw = gate_orientations[next_gate_ids][:, 2].unsqueeze(1)

        # Final observation vector (20 dimensions)
        obs = torch.cat(
            [
                curr_gate_rel_b,             # p_g^i
                lin_vel,                     # v_g^i
                drone_euler,
                ang_vel,                     # Ω
                rotor_cmds,                  # ω (throttle)
                next_gate_rel_b,             # p_g^{i+1}
                next_gate_yaw,               # ψ_g^{i+1}
            ],
            dim=-1,
        )

        return {"policy": obs}
     
    def _get_rewards(self) -> torch.tensor:
        """compute the rewards for all environments."""
        device = self.device
        num_envs = self.num_envs
        reward = torch.zeros(num_envs, 1, device=device)

        pk = self._robot.data.root_pos_w  # current position (n, 3)
        
        # store previous position if not exist
        if not hasattr(self, "prev_pos"):
            self.prev_pos = pk.clone()
        
        pk_prev = self.prev_pos
        
        # current target gate positions
        gate_ids = self.progress_buf  # index of current target gate
        gate_positions = torch.stack([torch.tensor(pos, device=device) for pos, _ in gate_features])
        pgk = gate_positions[gate_ids]  # (n,3)
        
        # ---------------------------
        # 2. progress reward
        # ---------------------------
        prev_dist = torch.norm(pk_prev - pgk, dim=1, keepdim=True)
        curr_dist = torch.norm(pk - pgk, dim=1, keepdim=True)
        progress_reward = prev_dist - curr_dist  # positive if drone moves closer
        
        # ---------------------------
        # 3. angular rate penalty
        # ---------------------------
        omega = self._robot.data.root_ang_vel_b  
        rate_penalty = 0.001 * torch.norm(omega, dim=1, keepdim=True)
        
        # ---------------------------
        # 4. base reward
        # ---------------------------
        reward = progress_reward - rate_penalty
        
        # ---------------------------
        # 5. collision / out-of-bounds
        # ---------------------------
        pos = pk
        out_of_bounds = (
            (pos[:, 0] < -5) | (pos[:, 0] > 5) |
            (pos[:, 1] < -5) | (pos[:, 1] > 5) |
            (pos[:, 2] < 0)  | (pos[:, 2] > 7)
        )
        collided = self._robot.data.root_pos_w[:,2] < 0.2 
        crashed = out_of_bounds | collided
        reward[crashed] = -10.0
        
        # ---------------------------
        # 7. update previous position
        # ---------------------------
        self.prev_pos = pk.clone()
        
        return reward
        
    def _get_dones(self) -> tuple[torch.tensor, torch.tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            height_too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.reset_height_threshold
            height_too_high = self._robot.data.root_pos_w[:, 2] > 200.0 

            died = torch.logical_or(height_too_low, height_too_high)
        

            return died, time_out

    def _reset_idx(self, env_ids: torch.tensor | none):
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

        num_starts = len(starting_position)

        for i, env_id in enumerate(env_ids):
            # 1) random selection among the starting positions
            start_id = torch.randint(0, num_starts, (1,), device=self.device).item()
            sx, sy, sz = starting_position[start_id]

            # 2) finding the closest gates near this starting positions
            min_dist = float("inf")
            closest_gate_id = 0
            for gate_id, (gate_pos, _) in enumerate(gate_features):
                gx, gy, gz = gate_pos
                dist = (sx - gx) ** 2 + (sy - gy) ** 2 # no need for the third dimensions
                if dist < min_dist:
                    min_dist = dist
                    closest_gate_id = gate_id

            # 3) orientation of the nearest gates
            _, gate_rot = gate_features[closest_gate_id]
            yaw = torch.tensor(gate_rot[2], device=self.device)
            qw = torch.cos(yaw / 2)
            qz = torch.sin(yaw / 2)
            spawn_quat = torch.tensor([qw, 0.0, 0.0, qz], device=self.device)

            # 5) assigner la position et l'orientation au drone
            root_pose[i, :3] = torch.tensor([sx, sy, sz], device=self.device)
            root_pose[i, 3:] = spawn_quat

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

