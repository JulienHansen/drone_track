# drone_dynamics.py
import math
import torch


class DroneDynamics:
    """
    Vectorized drone dynamics helper.

    - All tensors are batched with leading dim = num_envs.
    - Motor indexing convention used here (quad X):
        motor 0: front-left
        motor 1: front-right
        motor 2: rear-left
        motor 3: rear-right
      Adjust if your simulation uses a different ordering.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.device = device
        self.num_envs = num_envs

        # --- Randomized / per-env parameters (initialized in reset) ---
        self.mass = torch.empty(self.num_envs, 1, device=self.device)          # [N,1]
        self.J = torch.empty(self.num_envs, 3, device=self.device)            # [N,3] Jxx,Jyy,Jzz
        self.J_zz_rp = torch.empty(self.num_envs, 1, device=self.device)      # [N,1] rotor inertia proxy
        self.kl = torch.empty(self.num_envs, 1, device=self.device)          # [N,1] thrust coefficient
        self.kd = torch.empty(self.num_envs, 1, device=self.device)          # [N,1] drag/torque coefficient
        self.kx = torch.empty(self.num_envs, 1, device=self.device)          # aero drag x
        self.ky = torch.empty(self.num_envs, 1, device=self.device)          # aero drag y
        self.lb = torch.empty(self.num_envs, 1, device=self.device)          # rear arm length
        self.lf = torch.empty(self.num_envs, 1, device=self.device)          # front arm length
        self.theta_b = torch.empty(self.num_envs, 1, device=self.device)     # rear tilt angle (rad)
        self.theta_f = torch.empty(self.num_envs, 1, device=self.device)     # front tilt angle (rad)
        self.sigma_f = torch.empty(self.num_envs, 1, device=self.device)     # OU noise scale forces
        self.sigma_tau = torch.empty(self.num_envs, 1, device=self.device)   # OU noise scale torques
        self.max_thrust_to_weight = torch.empty(self.num_envs, 1, device=self.device)
        self.max_rate = torch.empty(self.num_envs, 3, device=self.device)
        self.max_prop_speed = torch.empty(self.num_envs, 1, device=self.device)

        # fixed constants
        self.g = torch.tensor(9.81, device=self.device)
        self.kappa = torch.full((self.num_envs, 1), 0.02, device=self.device)  # motor first-order time constant

        # persistent states
        self.prop_speeds = torch.zeros(self.num_envs, 4, device=self.device)   # current prop speeds (rad/s)
        self.f_res = torch.zeros(self.num_envs, 3, device=self.device)         # residual forces (OU)
        self.tau_res = torch.zeros(self.num_envs, 3, device=self.device)       # residual torques (OU)

        # OU params
        self.lambda_f = 1.0
        self.lambda_tau = 1.0

    def reset(self, env_ids):
        """
        Domain randomization / parameter initialization for the specified envs (env_ids can be a list or tensor).
        Use device-aware random sampling if you want randomized values; for now we use sensible defaults.
        """
        if len(env_ids) == 0:
            return

        # Example deterministic defaults (replace with torch sampling for real DR)
        self.mass[env_ids] = 0.795
        self.J[env_ids, 0] = 0.007
        self.J[env_ids, 1] = 0.007
        self.J[env_ids, 2] = 0.012

        self.J_zz_rp[env_ids] = 4e-4
        self.kl[env_ids] = 8.54e-6
        self.kd[env_ids] = 1.37e-6
        self.kx[env_ids] = 5e-5
        self.ky[env_ids] = 5e-5
        self.lb[env_ids] = 0.17
        self.lf[env_ids] = 0.17

        self.max_thrust_to_weight[env_ids] = 2.0
        self.max_rate[env_ids, :] = 12.85
        self.max_prop_speed[env_ids] = 838.0

        self.theta_b[env_ids] = 45.0 * (math.pi / 180.0)
        self.theta_f[env_ids] = 45.0 * (math.pi / 180.0)

        self.sigma_f[env_ids] = 0.0
        self.sigma_tau[env_ids] = 0.0

        # reset dynamic states
        self.prop_speeds[env_ids] = 0.0
        self.f_res[env_ids] = 0.0
        self.tau_res[env_ids] = 0.0

    def build_mixer_matrix(self):
        """
        Build per-env mixer matrices M such that:
            [ T_total; tau_x; tau_y; tau_z ] = M @ [f1,f2,f3,f4]
        where f_i are per-motor vertical thrust scalars (N).
        Returns:
            M: torch.Tensor of shape [N,4,4]
        Notes:
            - motor geometry must be consistent with compute_forces_and_torques.
            - motor indexing: 0:front-left, 1:front-right, 2:rear-left, 3:rear-right
        """
        N = self.num_envs
        device = self.device

        # Build per-motor positions r_i (N,4,3)
        # x_offsets: positive forward for front motors, negative for rear
        x_offsets = torch.stack([
            self.lf.squeeze(-1),    # m0 front-left
            self.lf.squeeze(-1),    # m1 front-right
           -self.lb.squeeze(-1),    # m2 rear-left (rear -> negative x)
           -self.lb.squeeze(-1)     # m3 rear-right
        ], dim=1)  # [N,4]

        # y offsets: left negative, right positive
        y_offsets = torch.stack([
           -self.lf.squeeze(-1),   # m0 front-left: left
            self.lf.squeeze(-1),   # m1 front-right: right
           -self.lb.squeeze(-1),   # m2 rear-left
            self.lb.squeeze(-1)    # m3 rear-right
        ], dim=1)  # [N,4]

        z_offsets = torch.zeros((N, 4), device=device)

        r = torch.stack([x_offsets, y_offsets, z_offsets], dim=2)  # [N,4,3]

        # thrust direction unit vectors d_i (tilt along x for front/rear)
        theta_f4 = self.theta_f.expand(-1, 4)
        theta_b4 = self.theta_b.expand(-1, 4)
        # theta per motor
        theta_per_motor = torch.stack([
            theta_f4[:, 0], theta_f4[:, 1], theta_b4[:, 2], theta_b4[:, 3]
        ], dim=1).unsqueeze(-1)  # [N,4,1]

        # x sign for front(+)/rear(-)
        x_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device).unsqueeze(0).unsqueeze(-1)  # [1,4,1]
        d_x = torch.sin(theta_per_motor) * x_sign  # [N,4,1]
        d_y = torch.zeros_like(d_x)                 # we keep y tilt = 0
        d_z = torch.cos(theta_per_motor)
        d = torch.cat([d_x, d_y, d_z], dim=2)      # [N,4,3]

        # Now build M elements:
        # row 0: total thrust T = sum_i f_i (since f_i are scalars along direction d_i, but we treat f_i as the scalar *along d_i*'s magnitude projection on body z)
        # To keep linearity in f_i (scalars meaning vertical thrust magnitude along d_i's magnitude),
        # we define f_i as the magnitude along the motor thrust vector, and T total is the z-component sum:
        # T = sum_i (f_i * d_i_z)
        # tau = sum_i (r_i x (f_i * d_i)) = sum_i f_i * (r_i x d_i)
        #
        # M[:,0,i] = d_i_z
        # M[:,1,i] = (r_i x d_i)_x
        # M[:,2,i] = (r_i x d_i)_y
        # M[:,3,i] = spin_sign_i * (kd/kl) * d_factor  -> but simpler: treat yaw map as moment per unit thrust via rotor drag:
        # For linear mapping we will write tau_z = sum_i sign_i * kd * omega_i^2.
        # But since f_i = kl * omega_i^2 and kl != 1, tau_z = sum_i sign_i * (kd/kl) * f_i.
        #
        # So M row 3 uses factor (sign_i * kd / kl)

        # compute r x d for each motor
        rx = r[:, :, 0:1]  # [N,4,1]
        ry = r[:, :, 1:2]
        rz = r[:, :, 2:3]

        dx = d[:, :, 0:1]
        dy = d[:, :, 1:2]
        dz = d[:, :, 2:3]

        # cross product r x d components
        cross_x = ry * dz - rz * dy  # [N,4,1]
        cross_y = rz * dx - rx * dz
        cross_z = rx * dy - ry * dx

        # row0: total thrust contribution per motor = d_z (z projection)
        row0 = dz.squeeze(-1)  # [N,4]

        # row1: tau_x contributions = cross_x.squeeze
        row1 = cross_x.squeeze(-1)  # [N,4]

        # row2: tau_y contributions = cross_y.squeeze
        row2 = cross_y.squeeze(-1)  # [N,4]

        # row3: tau_z (yaw) contribution from rotor drag mapped via (kd / kl) * sign
        # spin_sign typical quad-X: [1, -1, -1, 1]
        spin_sign = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device).unsqueeze(0)  # [1,4] -> [N,4]
        # kd/kl ratio: (N,1) -> expand
        ratio = (self.kd / torch.clamp(self.kl, min=1e-12)).expand(-1, 4)  # [N,4]
        row3 = spin_sign * ratio  # [N,4]

        # stack rows to build M: (N,4,4) where M[:,r,c] = row_r[:,c]
        M = torch.stack([row0, row1, row2, row3], dim=1)  # [N,4,4]
        # transpose to shape [N,4,4] where columns index motors: currently rows are correct orientation
        # But we want M @ f = w where f is [N,4,1]. Our M is currently shape [N,4,4] with M[row, col]=...
        # This is already the correct orientation.
        return M

    def compute_forces_and_torques(self, motors, lin_vel_b, dt):
        """
        motors: [N,4] prop speeds (rad/s) (target steady-state speeds from controller)
        lin_vel_b: [N,3] linear velocity in body frame
        dt: float timestep

        Returns:
            total_forces_b: [N,3]
            total_torques_b: [N,3]
        """
        N = self.num_envs
        device = self.device

        # --- motor dynamics (first order) ---
        omega_ss = motors
        dot_omega = (omega_ss - self.prop_speeds) / self.kappa  # [N,4]
        self.prop_speeds = self.prop_speeds + dot_omega * dt
        self.prop_speeds = torch.clamp(self.prop_speeds, min=0.0)
        omega2 = self.prop_speeds ** 2  # [N,4]

        # per-motor thrust scalars: f_i = kl * omega_i^2
        kl4 = self.kl.expand(-1, 4)
        thrust_per_motor = kl4 * omega2  # [N,4]

        # Build directions d_i and positions r exactly like build_mixer_matrix
        x_offsets = torch.stack([
            self.lf.squeeze(-1),
            self.lf.squeeze(-1),
           -self.lb.squeeze(-1),
           -self.lb.squeeze(-1)
        ], dim=1)  # [N,4]
        y_offsets = torch.stack([
           -self.lf.squeeze(-1),
            self.lf.squeeze(-1),
           -self.lb.squeeze(-1),
            self.lb.squeeze(-1)
        ], dim=1)  # [N,4]
        z_offsets = torch.zeros((N, 4), device=device)
        r = torch.stack([x_offsets, y_offsets, z_offsets], dim=2)  # [N,4,3]

        theta_f4 = self.theta_f.expand(-1, 4)
        theta_b4 = self.theta_b.expand(-1, 4)
        theta_per_motor = torch.stack([
            theta_f4[:, 0], theta_f4[:, 1], theta_b4[:, 2], theta_b4[:, 3]
        ], dim=1).unsqueeze(-1)  # [N,4,1]

        x_sign = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device).unsqueeze(0).unsqueeze(-1)
        d_x = torch.sin(theta_per_motor) * x_sign
        d_y = torch.zeros_like(d_x)
        d_z = torch.cos(theta_per_motor)
        d = torch.cat([d_x, d_y, d_z], dim=2)  # [N,4,3]

        # per-motor vector forces F_i = f_i * d_i
        thrusts = thrust_per_motor.unsqueeze(2) * d  # [N,4,3]

        # total prop force (sum)
        f_prop_b = thrusts.sum(dim=1)  # [N,3]

        # aerodynamic drag (simple)
        sum_omega = self.prop_speeds.sum(dim=1, keepdim=True)  # [N,1]
        f_aero_b = torch.zeros((N, 3), device=device)
        f_aero_b[:, 0:1] = -self.kx * lin_vel_b[:, 0:1] * sum_omega
        f_aero_b[:, 1:2] = -self.ky * lin_vel_b[:, 1:2] * sum_omega

        # OU residual forces
        noise_f = torch.randn_like(self.f_res)
        self.f_res = self.f_res + (-self.lambda_f * self.f_res * dt + self.sigma_f * math.sqrt(dt) * noise_f)

        total_forces_b = f_prop_b + f_aero_b + self.f_res  # [N,3]

        # torques: tau = sum_i (r_i x F_i)
        tau_from_thrust = torch.cross(r, thrusts, dim=2).sum(dim=1)  # [N,3]

        # yaw reaction torque from rotor drag: tau_z = sum sign_i * kd * omega^2
        spin_sign = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device).unsqueeze(0).expand(N, -1)  # [N,4]
        kd4 = self.kd.expand(-1, 4)
        tau_yaw_per_motor = spin_sign * (kd4 * omega2)  # [N,4]
        tau_yaw = torch.stack([torch.zeros(N, device=device),
                               torch.zeros(N, device=device),
                               tau_yaw_per_motor.sum(dim=1)], dim=1)  # [N,3]

        # motor reaction torque from rotor acceleration
        tau_mot_z = (self.J_zz_rp.expand(-1, 4) * (-dot_omega * spin_sign)).sum(dim=1)  # [N]
        tau_mot_b = torch.stack([torch.zeros(N, device=device),
                                 torch.zeros(N, device=device),
                                 tau_mot_z], dim=1)

        # OU residual torques
        noise_tau = torch.randn_like(self.tau_res)
        self.tau_res = self.tau_res + (-self.lambda_tau * self.tau_res * dt + self.sigma_tau * math.sqrt(dt) * noise_tau)

        total_torques_b = tau_from_thrust + tau_yaw + tau_mot_b + self.tau_res  # [N,3]

        return total_forces_b, total_torques_b

    def betaflight_rate_profile(
        self,
        rc_input,                         # shape: [N, 4]  [thrust_norm, roll, pitch, yaw] in [-1,1]
        rc_rate=torch.tensor([1.58, 1.55, 1.00]),
        super_rate=torch.tensor([0.73, 0.73, 0.73]),
        rc_expo=torch.tensor([0.30, 0.30, 0.30]),
        super_expo_active=True,
        limit=torch.tensor([2000.0, 2000.0, 2000.0])
    ):
        """
        Vectorized Betaflight-like rate profile.
        Returns: [N,4] = [total_thrust (N), p_dot_set (rad/s), q_dot_set, r_dot_set]
        - rc_input[:,0] is throttle in [-1,1]
        - rc_input[:,1:4] are roll/pitch/yaw sticks in [-1,1]
        """
        device = rc_input.device
        rc_rate = rc_rate.view(1, 3).to(device)
        super_rate = super_rate.view(1, 3).to(device)
        rc_expo = rc_expo.view(1, 3).to(device)
        limit = limit.view(1, 3).to(device)

        # shape rc stick part
        sticks = rc_input[:, 1:4]  # [N,3]

        # RC shaping (expo)
        expo_power = 3
        rc_input_shaped = sticks * (sticks.abs() ** expo_power) * rc_expo + sticks * (1 - rc_expo)  # [N,3]

        if super_expo_active:
            rc_factor = 1.0 / torch.clamp(1.0 - rc_input_shaped.abs() * super_rate, min=0.01)
            angular_vel = 200.0 * rc_rate * rc_input_shaped * rc_factor  # [N,3]
        else:
            angular_vel = (((rc_rate * 100.0) + 27.0) * rc_input_shaped / 16.0) / 4.1

        angular_vel = torch.clamp(angular_vel, -limit, limit)  # [N,3]

        # total thrust mapping: rc_input[:,0] in [-1,1] -> [0, max_thrust]
        total_thrust = (rc_input[:, 0].unsqueeze(1) + 1.0) * (self.max_thrust_to_weight * self.mass * self.g) / 2.0  # [N,1]

        return torch.cat([total_thrust, angular_vel], dim=1)  # [N,4]

