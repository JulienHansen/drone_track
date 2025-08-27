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
from isaaclab.markers import CUBOID_MARKER_CFG
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.core.utils.viewports import set_camera_view


@configclass
class TrackEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 1
    action_space = 4
    observation_space = 22
    state_space = 0
    moment_scale = 0.02
    debug_vis = True 
    reset_tracking_error_threshold = .5  
    reset_height_threshold = 0.2  


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

class TrackEnv(DirectRLEnv):
    cfg: TrackEnvCfg

    def __init__(self, cfg: TrackEnvCfg, render_mode: str | None = None, **kwargs):

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

        self._body_id = self._robot.find_bodies("body")[0]

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        # Randomization of the 3D lemniscate
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.8, 1.8, 1.], device=self.device),
            torch.tensor([3.2, 3.2, 1.5], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.8, device=self.device),
            torch.tensor(1.1, device=self.device)
        )

        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
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

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        self.target_pos[:] = self._compute_traj(self.future_traj_steps)
        self.rpos[:] = self.target_pos - self._robot.data.root_pos_w.unsqueeze(1)

        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)

        x = self._robot.data.root_pos_w[0]
        set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        obs = torch.cat(
            [
                self.rpos.flatten(start_dim=1),
                t,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance = torch.norm(self.rpos[:, 0], dim=-1)
        reward_pose = torch.exp(-1.2 * distance)

        tiltage = torch.abs(1 - self._robot.data.projected_gravity_b[:, 2])
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        effort = self._actions[:, 0]
        reward_effort = 0.02 * torch.exp(-effort)

        drone_vel_w = self._robot.data.root_lin_vel_w
        target_direction = F.normalize(self.target_pos[:, 1] - self.target_pos[:, 0], dim=-1)
        progress = torch.sum(drone_vel_w * target_direction, dim=-1)
        reward_progress = torch.relu(progress) * 0.1 
        reward = reward_pose + (reward_pose * reward_up) + reward_effort + reward_progress

        self._episode_sums["reward_pose"] += reward_pose
        self._episode_sums["reward_up"] += reward_up
        self._episode_sums["reward_effort"] += reward_effort
        if "reward_progress" not in self._episode_sums:
            self._episode_sums["reward_progress"] = torch.zeros_like(self._episode_sums["reward_pose"])
        self._episode_sums["reward_progress"] += reward_progress
        return reward

    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...

        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0.unsqueeze(1) + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * 1) # Hardcoded for now, should change later
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)

        target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        target_pos = vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self._terrain.env_origins[env_ids].unsqueeze(1) + target_pos + torch.tensor([0., 0., 2], device=self.device)


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            tracking_error = torch.norm(self.rpos[:, 0], dim=-1)
            tracking_error_exceeded = tracking_error > self.cfg.reset_tracking_error_threshold

            height_too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.reset_height_threshold
            height_too_high = self._robot.data.root_pos_w[:, 2] > 200.0 

            died = torch.logical_or(
                tracking_error_exceeded,
                torch.logical_or(height_too_low, height_too_high)
            )

            return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums.keys():
            avg_sum = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg_sum / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self.extras["log"] = extras
        self._robot.reset(env_ids)

        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w 
        self.traj_t0[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        t0 = self.traj_t0[env_ids]
        traj_c = self.traj_c[env_ids]
        traj_rot = self.traj_rot[env_ids]
        traj_scale = self.traj_scale[env_ids]
        origin = self._terrain.env_origins[env_ids]

        pos = lemniscate(t0, traj_c)
        pos = quat_rotate(traj_rot, pos)
        pos = pos * traj_scale
        pos = pos + origin
        pos[:, 2] += 2.0 
        spawn_pos = pos

        t0 = self.traj_t0[env_ids]
        traj_c = self.traj_c[env_ids]
        traj_rot = self.traj_rot[env_ids]
        traj_scale = self.traj_scale[env_ids]

        epsilon = 1e-2
        traj_pos_p = lemniscate(t0 + epsilon, traj_c)
        traj_pos_m = lemniscate(t0 - epsilon, traj_c)
        tangent = (traj_pos_p - traj_pos_m) / (2 * epsilon)


        tangent = quat_rotate(traj_rot, tangent)
        tangent = tangent * traj_scale
        forward = F.normalize(tangent, dim=-1)  

        global_up = torch.tensor([0., 0., 1.], device=self.device).expand_as(forward)
        right = F.normalize(torch.cross(global_up, forward, dim=-1), dim=-1)
        up = torch.cross(forward, right, dim=-1)

        rot_mat = torch.stack([forward, right, up], dim=-1) 
        rot = rotation_matrix_to_quaternion(rot_mat)

        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids] 

        root_pose = torch.cat([spawn_pos, rot], dim=1)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if env_ids[0] == 0:
            steps_to_draw = 200  
            t = torch.linspace(0, 2 * torch.pi, steps_to_draw, device=self.device) 

            env_id = env_ids[0]
            traj_c = self.traj_c[env_id]
            traj_rot = self.traj_rot[env_id]
            traj_scale = self.traj_scale[env_id]
            origin = self._terrain.env_origins[env_id]

            traj_points = lemniscate(t, traj_c.unsqueeze(0)).squeeze(0)  
            traj_points = quat_rotate(traj_rot.unsqueeze(0).expand(steps_to_draw, 4), traj_points)  
            traj_points = traj_points * traj_scale.unsqueeze(0)  
            traj_points = traj_points + origin.unsqueeze(0)  
            traj_points[:, 2] += 2

            point_list_0 = traj_points[:-1].tolist()
            point_list_1 = traj_points[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0)] * len(point_list_0)
            sizes = [1] * len(point_list_0)

            self.draw.clear_lines()
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:

            drone_marker_cfg = CUBOID_MARKER_CFG.copy()
            drone_marker_cfg.markers["cuboid"].size = (0.1,0.1,0.1)
            drone_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
            drone_marker_cfg.prim_path = "/Visuals/Command/drone_marker"
            self.drone_marker = VisualizationMarkers(drone_marker_cfg)
            self.drone_marker.set_visibility(True)
        else:

            self.drone_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        drone_marker_positions = self._robot.data.root_pos_w.clone()
        self.drone_marker.visualize(drone_marker_positions)


