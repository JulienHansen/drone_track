import torch
import math

class DroneDynamics:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs

        # --- Randomized Parameters (from Table 1) ---
        self.mass = torch.empty(self.num_envs, 1, device=self.device)
        self.J = torch.empty(self.num_envs, 3, device=self.device) # Jxx, Jyy, Jzz
        self.J_zz_rp = torch.empty(self.num_envs, 1, device=self.device) 
        self.kl = torch.empty(self.num_envs, 1, device=self.device)
        self.kd = torch.empty(self.num_envs, 1, device=self.device)
        self.kx = torch.empty(self.num_envs, 1, device=self.device)
        self.ky = torch.empty(self.num_envs, 1, device=self.device)
        self.lb = torch.empty(self.num_envs, 1, device=self.device)
        self.lf = torch.empty(self.num_envs, 1, device=self.device)
        self.theta_b = torch.empty(self.num_envs, 1, device=self.device)
        self.theta_f = torch.empty(self.num_envs, 1, device=self.device)
        self.sigma_f = torch.empty(self.num_envs, 1, device=self.device)
        self.sigma_tau = torch.empty(self.num_envs, 1, device=self.device)
        self.max_thrust_to_weight = torch.empty(self.num_envs, 1, device=self.device)
        self.g   = torch.tensor(9.81, device=self.device)
        self.max_rate = torch.empty(self.num_envs, 3, device=self.device) # Jxx, Jyy, Jzz
        self.max_prop_speed = torch.empty(self.num_envs, 1, device=self.device) # Jxx, Jyy, Jzz
        self.prop_speeds = torch.empty(self.num_envs, 4, device=self.device)
        # --- Fixed Parameters ---
        self.kappa = torch.full((self.num_envs, 1), 0.02, device=self.device)

        self.f_res = torch.zeros(self.num_envs, 3, device=self.device)
        self.tau_res = torch.zeros(self.num_envs, 3, device=self.device)

        # Ornstein-Uhlenbeck process parameters -> not specified so TODO: change
        self.lambda_f = 1.0
        self.lambda_tau = 1.0


    def reset(self, env_ids):
        """Resets the parameters of specified environment (also perform domain randomization here)."""
        if len(env_ids) == 0:
            return

        # --- Domain Randomization --- 
        self.mass[env_ids] = 0.795 #torch.empty(len(env_ids), 1, device=self.device).uniform_(0.15, 1.4)
        self.J[env_ids, 0] = 0.007 #torch.empty(len(env_ids), device=self.device).uniform_(3e-3, 6e-2) # Jxx
        self.J[env_ids, 1] = 0.007 #torch.empty(len(env_ids), device=self.device).uniform_(3e-3, 6e-2) # Jyy
        self.J[env_ids, 2] = 0.012 #torch.empty(len(env_ids), device=self.device).uniform_(6e-3, 2e-1) # Jzz
        
        # TODO: J_zz_rp in Eq. 9 is not randomized normal ?.
        self.J_zz_rp[env_ids] = 4e-4  #torch.empty(len(env_ids), 1, device=self.device).uniform_(4e-4, 3e-3)
        self.kl[env_ids] = 8.54e-6 #torch.empty(len(env_ids), 1, device=self.device).uniform_(4e-7, 8e-5)
        self.kd[env_ids] = 1.37e-6 #torch.empty(len(env_ids), 1, device=self.device).uniform_(8e-8, 4e-6) 
        self.kx[env_ids] = 5e-5 #torch.empty(len(env_ids), 1, device=self.device).uniform_(2e-5, 1e-4)
        self.ky[env_ids] = 5e-5 #torch.empty(len(env_ids), 1, device=self.device).uniform_(2e-5, 1e-4)
        self.lb[env_ids] = 0.17 #torch.empty(len(env_ids), 1, device=self.device).uniform_(0.07, 0.16)
        self.lf[env_ids] = 0.17 #torch.empty(len(env_ids), 1, device=self.device).uniform_(0.07, 0.16)

        self.max_thrust_to_weight[env_ids] = 2.0 #torch.empty(len(env_ids), 1, device=self.device).uniform_(4.0, 15.0)
        self.max_rate[env_ids, 0] = 12.85 
        self.max_rate[env_ids, 1] = 12.85 
        self.max_rate[env_ids, 2] = 12.85 
        self.max_prop_speed[env_ids] = 838.0 #torch.empty(len(env_ids), 1, device=self.device).uniform_(400.0, 838.0)

        self.theta_b[env_ids] = 45 * (math.pi / 180.0) #torch.empty(len(env_ids), 1, device=self.device).uniform_(35, 55) * (math.pi / 180.0)
        self.theta_f[env_ids] = 45 * (math.pi / 180.0) #torch.empty(len(env_ids), 1, device=self.device).uniform_(35, 55) * (math.pi / 180.0)

        self.sigma_f[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.0, 0.6)
        self.sigma_tau[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.0, 0.03)

        
        

        # --- Persistent states ---
        self.prop_speeds[env_ids] = 0.0
        self.f_res[env_ids] = 0.0
        self.tau_res[env_ids] = 0.0


    def compute_forces_and_torques(self, motors, lin_vel_b, dt):
        """
        Computes total forces and torques in the body frame.

        Args:
            actions (torch.Tensor): Normalized motor commands [0, 1] of shape (num_envs, 4).
            lin_vel_b (torch.Tensor): Linear velocity in the body frame of shape (num_envs, 3).
            dt (float): Simulation time step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - total_forces_b (torch.Tensor): Total forces in the body frame (num_envs, 3).
                - total_torques_b (torch.Tensor): Total torques in the body frame (num_envs, 3).
        """

        #TODO : Need to check if the actions passed are between 0 and 1 or -1 and 1 it should change the function
        # Needs to check every equation, especially the one for the Torques
        # Petite zone de test en dessous afin de tester tout ca.


        # --- 1. Motor Dynamics (Eq. 5) --- 
        omega_ss = motors 
        dot_omega = (omega_ss - self.prop_speeds) / self.kappa
        self.prop_speeds += dot_omega * dt
        self.prop_speeds = torch.clamp(self.prop_speeds, min=0.0)
        
        prop_speeds_sq = self.prop_speeds**2


        # Propeller Thrust (Eq. 6) 
        thrust_magnitude = torch.sum(self.kl * prop_speeds_sq, dim=1)
        f_prop_b = torch.zeros(self.num_envs, 3, device=self.device)
        f_prop_b[:, 2] = thrust_magnitude

        # Aerodynamic Drag (Eq. 7) [cite: 60, 61]
        sum_omega = torch.sum(self.prop_speeds, dim=1, keepdim=True)
        f_aero_b = torch.zeros(self.num_envs, 3, device=self.device)
        f_aero_b[:, 0:1] = -self.kx * lin_vel_b[:, 0].unsqueeze(-1) * sum_omega
        f_aero_b[:, 1:2] = -self.ky * lin_vel_b[:, 1].unsqueeze(-1) * sum_omega # Assuming ky for y-axis

        # Residual Forces (Eq. 10) 
        noise_f = torch.randn_like(self.f_res)
        self.f_res += -self.lambda_f * self.f_res * dt + self.sigma_f * math.sqrt(dt) * noise_f
        

        
        # Propeller Torque (Eq. 8) 
        tau_prop_b = torch.zeros(self.num_envs, 3, device=self.device)
        # completement pas sur faudra verifier LOL
        tau_prop_b[:, 0] = ( -self.lb * torch.sin(self.theta_b) * self.kl * prop_speeds_sq[:, 0].unsqueeze(1)
                             -self.lf * torch.sin(self.theta_f) * self.kl * prop_speeds_sq[:, 1].unsqueeze(1)
                             +self.lb * torch.sin(self.theta_b) * self.kl * prop_speeds_sq[:, 2].unsqueeze(1)
                             +self.lf * torch.sin(self.theta_f) * self.kl * prop_speeds_sq[:, 3].unsqueeze(1) ).squeeze()

        tau_prop_b[:, 1] = ( +self.lb * torch.cos(self.theta_b) * self.kl * prop_speeds_sq[:, 0].unsqueeze(1)
                             -self.lf * torch.cos(self.theta_f) * self.kl * prop_speeds_sq[:, 1].unsqueeze(1)
                             +self.lb * torch.cos(self.theta_b) * self.kl * prop_speeds_sq[:, 2].unsqueeze(1)
                             -self.lf * torch.cos(self.theta_f) * self.kl * prop_speeds_sq[:, 3].unsqueeze(1) ).squeeze()
        
        
        tau_prop_b[:, 2] = self.kd.squeeze() * (prop_speeds_sq[:, 0] - prop_speeds_sq[:, 1] - prop_speeds_sq[:, 2] + prop_speeds_sq[:, 3])

        # Motor Reaction Torque (Eq. 9) 
        tau_mot_b = torch.zeros(self.num_envs, 3, device=self.device)
        tau_mot_b[:, 2] = self.J_zz_rp.squeeze() * (-dot_omega[:, 0] + dot_omega[:, 1] - dot_omega[:, 2] + dot_omega[:, 3])

        # Residual Torques (Eq. 11) 
        noise_tau = torch.randn_like(self.tau_res)
        self.tau_res += -self.lambda_tau * self.tau_res * dt + self.sigma_tau * math.sqrt(dt) * noise_tau

        # --- A la fin faut sommer pour avoir le thrust and thorques resultant ---
        total_forces_b = f_prop_b + f_aero_b + self.f_res
        total_torques_b = tau_prop_b + tau_mot_b + self.tau_res

        return total_forces_b, total_torques_b
    

    def betaflight_rate_profile(
        self,
        rc_input,                         # shape: [N, 4]
        rc_rate=torch.tensor([1.58, 1.55, 1.00]),
        super_rate=torch.tensor([0.73, 0.73, 0.73]),
        rc_expo=torch.tensor([0.30, 0.30, 0.30]),
        super_expo_active=True,
        limit=torch.tensor([2000.0, 2000.0, 2000.0])
    ):
        """
        Fully vectorized Betaflight rate profile over [N, 3] RC input.
        Each row of rc_input is a 3D command (roll, pitch, yaw).
        """
        
        rc_rate = rc_rate.view(1, 3).to(rc_input.device)
        super_rate = super_rate.view(1, 3).to(rc_input.device)
        rc_expo = rc_expo.view(1, 3).to(rc_input.device)
        limit = limit.view(1, 3).to(rc_input.device)

        # RC Rate > 2 shaping
        rc_rate = torch.where(rc_rate > 2, rc_rate + (rc_rate - 2) * 14.54, rc_rate)

        # Expo shaping
        expo_power = 3
        rc_input_shaped = rc_input[:, 1:4] * (rc_input[:, 1:4].abs() ** expo_power) * rc_expo + rc_input[:, 1:4] * (1 - rc_expo)

        # Super Expo shaping
        if super_expo_active:
            rc_factor = 1.0 / torch.clamp(1.0 - rc_input_shaped.abs() * super_rate, 0.01, 1.0)
            angular_vel = 200 * rc_rate * rc_input_shaped * rc_factor
        else:
            angular_vel = (((rc_rate * 100) + 27) * rc_input_shaped / 16.0) / 4.1
        
        total_thrust = (rc_input[:, 0].unsqueeze(1) + 1) * (self.max_thrust_to_weight * self.mass * float(self.g)) / 2  # [N]

        angular_vel = torch.clamp(angular_vel, -limit, limit)
        angular_vel = angular_vel.squeeze(1)   

        return torch.cat([total_thrust, angular_vel], dim=1)   


    def step_body_rates(self, torques_b, omega_b, dt):
        """
        Intègre uniquement les équations de rotation (body rates).
        
        Args:
            torques_b (torch.Tensor): [N, 3] couples totaux dans le repère corps
            omega_b (torch.Tensor): [N, 3] body rates actuels [p,q,r]
            dt (float): pas de temps
        
        Returns:
            new_omega_b (torch.Tensor): [N, 3] body rates mis à jour
        """
    
        # Moments d’inertie [N,3]
        Jx, Jy, Jz = self.J[:, 0], self.J[:, 1], self.J[:, 2]

        # J * omega
        J_omega = torch.stack([
            Jx * omega_b[:, 0],
            Jy * omega_b[:, 1],
            Jz * omega_b[:, 2]
        ], dim=1)

        # Terme gyroscopique ω × (Jω)
        cross_tau = torch.cross(omega_b, J_omega, dim=1)

        # Accélération angulaire
        domega_b = torch.zeros_like(omega_b)
        domega_b[:, 0] = (torques_b[:, 0] - cross_tau[:, 0]) / Jx
        domega_b[:, 1] = (torques_b[:, 1] - cross_tau[:, 1]) / Jy
        domega_b[:, 2] = (torques_b[:, 2] - cross_tau[:, 2]) / Jz

        # Intégration d’Euler
        new_omega_b = omega_b + domega_b * dt

        return new_omega_b



if __name__ == "__main__":
    dynamic = DroneDynamics()





