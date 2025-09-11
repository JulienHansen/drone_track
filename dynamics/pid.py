import os
import torch



class PIDController3D:
    def __init__(self, Kp, Ki, Kd, dt, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.Kp = Kp.to(device)
        self.Ki = Ki.to(device)
        self.Kd = Kd.to(device)
        if isinstance(dt, (int, float)):
            self.dt = torch.tensor(dt, device=device)
        else:
            self.dt = dt.to(device)
        self.device = device
        self.integral = None
        self.prev_error = None
        self.num_envs = None

    def _init_if_needed(self, target):
        if self.integral is None or self.prev_error is None:
            self.num_envs = target.shape[0]
            self.integral = torch.zeros((self.num_envs, 3), device=self.device)
            self.prev_error = torch.zeros((self.num_envs, 3), device=self.device)

    def forward(self, target, measurement):
        """
        Args:
            target: Tensor of shape [num_envs, 3]
            measurement: Tensor of shape [num_envs, 3]
        """
        self._init_if_needed(target)
        error = target - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        return (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset internal state for selected environments. If env_ids is None,
        reset all environments.
        """
        if self.integral is None or self.prev_error is None:
            return  # Nothing to reset

        if env_ids is None or (len(env_ids) == self.num_envs):
            env_ids = slice(None)  # Equivalent to [:]

        self.integral[env_ids] = 0.0
        self.prev_error[env_ids] = 0.0
