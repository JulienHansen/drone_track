# PIDController.py
import torch


class PIDController:
    """
    Cascade controller that:
      - Accepts setpoint [T_cmd (N), p_cmd (rad/s), q_cmd, r_cmd]
      - Uses thrust PID (on total force) and rate PID (on body rates) to compute:
          desired total thrust (N) and desired torques (Nm)
      - Uses dynamics mixer matrix M from DroneDynamics to compute per-motor thrusts:
          f = pinv(M) @ [T_cmd, tau_x, tau_y, tau_z]
      - Converts thrust -> prop speeds omega via omega = sqrt(f / kl)
      - Returns prop speeds (rad/s) to be passed to dynamics
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.device = device
        self.num_envs = num_envs

        # --- Gains (start conservative; tune in-sim) ---
        # Thrust PID (N)
        self.kp_thrust = torch.full((1,), 10.0, device=self.device)
        self.ki_thrust = torch.full((1,), 0.5, device=self.device)
        self.kd_thrust = torch.full((1,), 1.0, device=self.device)

        # Rate P gains (for p,q,r) -> produce torque proxies
        self.kp_rate = torch.tensor([0.03, 0.03, 0.015], device=self.device)  # [3]

        # integrators and previous errors (batched)
        self.integral = torch.zeros((self.num_envs, 4), device=self.device)  # [T, p, q, r]
        self.prev_err = torch.zeros((self.num_envs, 4), device=self.device)

        # dynamics placeholders (will be set in reset)
        # these must be provided from DroneDynamics.reset call
        self.kl = torch.ones(self.num_envs, 1, device=self.device)
        self.kd = torch.ones(self.num_envs, 1, device=self.device)
        self.J = torch.ones(self.num_envs, 3, device=self.device)
        self.lb = torch.ones(self.num_envs, 1, device=self.device)
        self.lf = torch.ones(self.num_envs, 1, device=self.device)
        self.theta_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.theta_f = torch.zeros(self.num_envs, 1, device=self.device)
        self.spin_sign = torch.tensor([1.0, -1.0, -1.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # per-env maximums
        self.max_thrust_to_weight = torch.ones(self.num_envs, 1, device=self.device)
        self.mass = torch.ones(self.num_envs, 1, device=self.device)
        self.g = torch.full((self.num_envs, 1), 9.81, device=self.device)
        self.max_prop_speed = torch.full((self.num_envs, 1), 1000.0, device=self.device)

        # store last M and pseudo-inverse for speed
        self.M = None          # [N,4,4]
        self.M_pinv = None     # [N,4,4]

        # stored setpoint
        self.setpoint = torch.zeros(self.num_envs, 4, device=self.device)  # [N,4] [T, p,q,r]

    def reset(self, env_ids,
              kl, kd, J, lb, lf, theta_b, theta_f,
              max_thrust_to_weight, mass, max_prop_speed):
        """
        Provide dynamics params from DroneDynamics so we can build the mixer.
        env_ids: list/tensor of env indices to reset
        """
        # assign dynamics params
        self.kl[env_ids] = kl[env_ids].to(self.device)
        self.kd[env_ids] = kd[env_ids].to(self.device)
        self.J[env_ids] = J[env_ids].to(self.device)
        self.lb[env_ids] = lb[env_ids].to(self.device)
        self.lf[env_ids] = lf[env_ids].to(self.device)
        self.theta_b[env_ids] = theta_b[env_ids].to(self.device)
        self.theta_f[env_ids] = theta_f[env_ids].to(self.device)
        self.max_thrust_to_weight[env_ids] = max_thrust_to_weight[env_ids].to(self.device)
        self.mass[env_ids] = mass[env_ids].to(self.device)
        self.max_prop_speed[env_ids] = max_prop_speed[env_ids].to(self.device)

        # reset integrators/prev errors for those envs
        self.integral[env_ids] = 0.0
        self.prev_err[env_ids] = 0.0

        # Build M and M_pinv for each env (vectorized)
        self.build_mixer_matrix_and_pinv(env_ids)

    def build_mixer_matrix_and_pinv(self, env_ids=None):
        """
        Build per-env mixer matrix M and pseudo-inverse M_pinv given current dynamics params.
        M maps f (per-motor thrust scalars) to [T, tau_x, tau_y, tau_z].
        We compute M_pinv with torch.linalg.pinv for numerical stability.
        """
        N = self.num_envs
        device = self.device

        # construct positions r and directions d similarly to DroneDynamics.build_mixer_matrix
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

        rx = r[:, :, 0:1]
        ry = r[:, :, 1:2]
        rz = r[:, :, 2:3]
        dx = d[:, :, 0:1]
        dy = d[:, :, 1:2]
        dz = d[:, :, 2:3]

        cross_x = ry * dz - rz * dy
        cross_y = rz * dx - rx * dz
        # cross_z = rx*dy - ry*dx  (not used)

        row0 = dz.squeeze(-1)   # [N,4]
        row1 = cross_x.squeeze(-1)
        row2 = cross_y.squeeze(-1)
        spin_sign = torch.tensor([1.0, -1.0, -1.0, 1.0], device=device).unsqueeze(0)  # [1,4]
        ratio = (self.kd / torch.clamp(self.kl, min=1e-12)).expand(-1, 4)  # [N,4]
        row3 = spin_sign * ratio

        M = torch.stack([row0, row1, row2, row3], dim=1)  # [N,4,4]

        # compute pseudo-inverse per env
        # torch.linalg.pinv supports batched matrices
        M_pinv = torch.linalg.pinv(M)  # [N,4,4]

        self.M = M
        self.M_pinv = M_pinv

    def reset_setpoint(self, setpoint):
        """
        setpoint: [N,4] tensor = [T_des (N), p_des (rad/s), q_des, r_des]
        """
        self.setpoint = setpoint

    def compute(self, process_variable, dt):
        """
        process_variable: [N,4] = [measured_total_force_z (N), p,q,r (rad/s)]
        returns: omega [N,4] prop speeds (rad/s) clamped to [0, max_prop_speed]
        """
        N = self.num_envs
        device = self.device

        measured_force = process_variable[:, 0].unsqueeze(1)  # [N,1]
        measured_rates = process_variable[:, 1:4]             # [N,3]

        # desired setpoints
        desired_force = self.setpoint[:, 0].unsqueeze(1)      # [N,1]
        desired_rates = self.setpoint[:, 1:4]                 # [N,3]

        # --- Thrust PID (N) ---
        err_thrust = desired_force - measured_force         # [N,1]
        self.integral[:, 0:1] += err_thrust * dt
        d_thrust = (err_thrust - self.prev_err[:, 0:1]) / dt

        T_p = self.kp_thrust.to(device) * err_thrust
        T_i = self.ki_thrust.to(device) * self.integral[:, 0:1]
        T_d = self.kd_thrust.to(device) * d_thrust

        T_cmd = T_p + T_i + T_d  # [N,1]

        # clamp to feasible total thrust [0, max_total_thrust]
        max_total_thrust = (self.max_thrust_to_weight * self.mass * self.g)
        T_cmd = torch.clamp(T_cmd, min=torch.zeros_like(T_cmd), max=max_total_thrust)

        # --- Rate P controller -> desired torques ---
        rate_err = desired_rates - measured_rates  # [N,3]
        # torque proxies: τ_des = kp_rate * rate_err * J (gives Nm scaling)
        kp_rate = self.kp_rate.to(device).unsqueeze(0)  # [1,3]
        tau_des = kp_rate * (self.J * rate_err)        # [N,3]

        # compose wrench vector per env: w = [T_cmd, tau_x, tau_y, tau_z]
        w = torch.cat([T_cmd, tau_des], dim=1)  # [N,4]

        # Build M_pinv if not present (safety)
        if self.M_pinv is None:
            self.build_mixer_matrix_and_pinv()

        # Solve for per-motor thrusts f (N,4) using pseudo-inverse: f = M_pinv @ w
        # Need w as [N,4,1] for batched matmul
        w_col = w.unsqueeze(-1)  # [N,4,1]
        f = torch.matmul(self.M_pinv, w_col).squeeze(-1)  # [N,4]

        # Enforce non-negative thrusts (no negative thrust)
        f = torch.clamp(f, min=torch.zeros_like(f))

        # Convert thrust -> omega: f = kl * omega^2  => omega = sqrt( f / kl )
        kl4 = self.kl.expand(-1, 4)  # [N,4]
        omega = torch.sqrt(torch.clamp(f / torch.clamp(kl4, min=1e-12), min=0.0))

        # clamp to prop speed limits
        omega = torch.clamp(omega, min=torch.zeros_like(omega), max=self.max_prop_speed.expand(-1, 4))

        # update prev_err
        self.prev_err[:, 0:1] = err_thrust
        self.prev_err[:, 1:4] = rate_err

        return omega

