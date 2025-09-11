import torch
import math
from pid import * 

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
        
        # --- Fixed Parameters ---
        self.kappa = torch.full((self.num_envs, 1), 0.01, device=self.device)
        self.max_omega = 1600.0 # TODO: Aucune idee de la valeur, j'ai un peu chercher mais pas sur 
        # TODO: This needs to be modified 

        self.prop_speeds = torch.zeros(self.num_envs, 4, device=self.device)
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
                self.mass[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.6, 1.4)
                self.J[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(3e-3, 1.2e-2) # Jxx
                self.J[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(3e-3, 1.2e-2) # Jyy
                self.J[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(6e-3, 2e-2) # Jzz
                
                # TODO: J_zz_rp in Eq. 9 is not randomized normal ?.
            self.J_zz_rp[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(6e-5, 1e-4)
            
            self.kl[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(5e-6, 1.5e-5)
            self.kd[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(8e-8, 4.5e-7) 
            self.kx[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.03, 0.2)
            self.ky[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.03, 0.2)
            self.lb[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.11, 0.17)
            self.lf[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.11, 0.17)

            self.theta_b[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(15, 75) * (math.pi / 180.0)
            self.theta_f[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(15, 75) * (math.pi / 180.0)

            self.sigma_f[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.0, 0.6)
            self.sigma_tau[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(0.0, 0.03)

            # --- Persistent states ---
            self.prop_speeds[env_ids] = 0.0
            self.f_res[env_ids] = 0.0
            self.tau_res[env_ids] = 0.0

        def pre_forces_compute(self, actions):
            
        desired_attitude = betaflight(actions)
        pid = New
        omega_ss = pid










    def compute_forces_and_torques(self, actions, lin_vel_b, dt):
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
        omega_ss = actions * self.max_omega
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
        f_aero_b[:, 0] = -self.kx * lin_vel_b[:, 0].unsqueeze(-1) * sum_omega
        f_aero_b[:, 1] = -self.ky * lin_vel_b[:, 1].unsqueeze(-1) * sum_omega # Assuming ky for y-axis

        # Residual Forces (Eq. 10) 
        noise_f = torch.randn_like(self.f_res)
        self.f_res += -self.lambda_f * self.f_res * dt + self.sigma_f * math.sqrt(dt) * noise_f
        

        
        # Propeller Torque (Eq. 8) 
        tau_prop_b = torch.zeros(self.num_envs, 3, device=self.device)
        # completement pas sur faudra verifier LOL
        tau_prop_b[:, 0] = ( -self.lb * torch.sin(self.theta_b) * self.kl * prop_speeds_sq[:, 0]
                             -self.lf * torch.sin(self.theta_f) * self.kl * prop_speeds_sq[:, 1]
                             +self.lb * torch.sin(self.theta_b) * self.kl * prop_speeds_sq[:, 2]
                             +self.lf * torch.sin(self.theta_f) * self.kl * prop_speeds_sq[:, 3] ).squeeze()

        tau_prop_b[:, 1] = ( +self.lb * torch.cos(self.theta_b) * self.kl * prop_speeds_sq[:, 0]
                             -self.lf * torch.cos(self.theta_f) * self.kl * prop_speeds_sq[:, 1]
                             +self.lb * torch.cos(self.theta_b) * self.kl * prop_speeds_sq[:, 2]
                             -self.lf * torch.cos(self.theta_f) * self.kl * prop_speeds_sq[:, 3] ).squeeze()
        
        tau_prop_b[:, 2] = self.kd.squeeze() * (prop_speeds_sq[:, 0] - prop_speeds_sq[:, 1] + prop_speeds_sq[:, 2] - prop_speeds_sq[:, 3])

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
    



if __name__ == "__main__":
    dynamic = DroneDynamics()





