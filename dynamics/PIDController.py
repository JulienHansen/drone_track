import torch

class PIDController:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.kp=torch.tensor([400.0, 450.0, 450.0, 450.0], device=self.device)   # thrust, roll, pitch, yaw
        self.ki=torch.tensor([400.0, 70.0, 70.0, 200.0], device=self.device)
        self.kd=torch.tensor([0.0, 10.0, 10.0, 25.0], device=self.device)
        self.kff = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        #self.Dmax = torch.tensor([0.0, 0.0, 0.0, 100.0], device=self.device)
        self.setpoint = torch.zeros(self.num_envs, 4, device=self.device)  # thrust, roll, pitch, yaw
        self.previous_errors = torch.zeros(self.num_envs, 4, device=self.device)
        self.integral = torch.zeros(self.num_envs, 4, device=self.device)
        self.max_thrust_to_weight  = torch.zeros(self.num_envs, 1, device=self.device)

        self.max_rate  = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_prop_speed = torch.zeros(self.num_envs, 1, device=self.device)
        self.mass = torch.zeros(self.num_envs, 1, device=self.device)
        self.g = torch.full((self.num_envs, 1), 9.81, device=self.device)

    def reset_setpoint(self, setpoint):
        self.setpoint = setpoint
    
    def reset(self, env_ids, max_thrust_to_weight, max_rate, max_prop_speed, mass):
        self.max_thrust_to_weight[env_ids] = max_thrust_to_weight[env_ids].to(self.device)
        self.max_rate[env_ids] = max_rate[env_ids].to(self.device)
        self.max_prop_speed[env_ids] = max_prop_speed[env_ids].to(self.device)
        self.mass[env_ids] = mass[env_ids].to(self.device)
        self.integral[env_ids] = 0
        self.previous_errors[env_ids] = 0


    def compute(self, process_variable, dt=0.002):
            
            """ Process state"""
            thrust_acc = process_variable[:, 0].unsqueeze(1) / self.mass # (T/m)
            thrust_acc = torch.clamp(thrust_acc, min=torch.zeros(self.num_envs, 1, device=self.device), max= self.max_thrust_to_weight  * self.g) # T/m in [0, T_max/m]
            rates = process_variable[:, 1:4] 
            rates = torch.clamp(rates, min=-self.max_rate, max=self.max_rate) # w in [-w_max, w_max]
            bounded_variables = torch.cat([thrust_acc, rates], dim=1)
            
            """ Reference setpoint """
            
            set_thrust_acc = self.setpoint[:, 0].unsqueeze(1) / self.mass # (T/m)
            set_thrust_acc = torch.clamp(set_thrust_acc, min=torch.zeros(self.num_envs, 1, device=self.device), max= self.max_thrust_to_weight  * self.g) # T/m in [0, T_max/m]
            set_rates = self.setpoint[:, 1:4] 
            set_rates = torch.clamp(set_rates, min=-self.max_rate, max=self.max_rate) # w in [-w_max, w_max]
            bounded_reference = torch.cat([set_thrust_acc, set_rates], dim=1)

    
            errors = bounded_reference - bounded_variables  # [N, 4] (thrust, roll, pitch, yaw)


            self.integral += errors * dt

            P = self.kp * errors
            I = self.ki * self.integral
            D = self.kd * (errors - self.previous_errors) / dt
            FF = self.kff * bounded_reference # Feedforward term

            pid_output = P + I + D + FF  # [N, 4] (thrust, roll, pitch, yaw)

            thrust_out = pid_output[:, 0]
            roll_out   = pid_output[:, 1]
            pitch_out  = pid_output[:, 2]
            yaw_out    = pid_output[:, 3]

            # --- Mixer (quad X) ---
            m1 = thrust_out + pitch_out - roll_out + yaw_out
            m2 = thrust_out - pitch_out - roll_out - yaw_out
            m3 = thrust_out + pitch_out + roll_out - yaw_out
            m4 = thrust_out - pitch_out + roll_out + yaw_out

            motors = torch.stack([m1, m2, m3, m4], dim=1)  # [N, 4]

            motors = torch.clamp(motors, min=torch.zeros(self.num_envs, 4, device=self.device), max=self.max_prop_speed.expand_as(motors))  # rad/s
            self.previous_errors = errors

            return motors  # rad/s