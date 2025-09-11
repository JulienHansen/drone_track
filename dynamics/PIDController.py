import torch

class PIDController:
    def __init__(self, num_envs, device, setpoint, max_thrust_to_weight=8.17, max_rate=[12.85, 12.85, 12.85], max_prop_speed= 838.0, FPV_mass=0.716, g=9.81):
        self.device = device
        self.num_envs = num_envs
        self.kp=torch.tensor([400.0, 450.0, 450.0, 450.0], device=self.device)   # thrust, roll, pitch, yaw
        self.ki=torch.tensor([400.0, 70.0, 70.0, 200.0], device=self.device)
        self.kd=torch.tensor([0.0, 10.0, 10.0, 25.0], device=self.device)
        self.kff = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        #self.dmax = torch.tensor([0.0, 0.0, 0.0, 100.0], device=self.device)
        self.setpoint = setpoint
        self.previous_errors = torch.zeros(self.num_envs, 4, device=self.device)
        self.integral = torch.zeros(self.num_envs, 4, device=self.device)
        self.max_thrust_to_weight = torch.tensor(max_thrust_to_weight, device=self.device) # N
        self.max_rate = torch.tensor(max_rate, device=self.device) # rad/s
        self.max_prop_speed = torch.tensor(max_prop_speed, device=self.device) # rad/s
        self.FPV_mass = torch.tensor(FPV_mass, device=self.device) # kg
        self.g = torch.tensor(g, device=self.device) # kg

    def compute(self, process_variable, dt=0.002):
            
            """ Process state"""
            thrust_acc = process_variable[:, 0] / self.FPV_mass # (T/m)
            thrust_acc = torch.clamp(thrust_acc, min=0.0, max= self.max_thrust_to_weight  * self.g) # T/m in [0, T_max/m]
            rates = process_variable[:, 1:4] 
            rates = torch.clamp(rates, min=-self.max_rate, max=self.max_rate) # w in [-w_max, w_max]
            bounded_variables = torch.cat([thrust_acc.unsqueeze(1), rates], dim=1)

            #print("bounded_variables:", bounded_variables)

            """ Reference setpoint """

            set_thrust_acc = self.setpoint[:, 0] / self.FPV_mass # (T/m)
            set_thrust_acc = torch.clamp(set_thrust_acc, min=0.0, max= self.max_thrust_to_weight  * self.g) # T/m in [0, T_max/m]
            set_rates = self.setpoint[:, 1:4] 
            set_rates = torch.clamp(set_rates, min=-self.max_rate, max=self.max_rate) # w in [-w_max, w_max]
            bounded_reference = torch.cat([set_thrust_acc.unsqueeze(1), set_rates], dim=1)

            #print("bounded_reference:", bounded_reference)
    
            errors = bounded_reference - bounded_variables  # [N, 4] (thrust, roll, pitch, yaw)

            print('errors:', errors)

            self.integral += errors * dt

            P = self.kp * errors
            I = self.ki * self.integral
            D = self.kd * (errors - self.previous_errors) / dt
            FF = self.kff * bounded_reference

            print('P:', P)
            print('I:', I)
            print('D:', D)

            #D = torch.clamp(D, -self.dmax, self.dmax)

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

            motors = torch.clamp(motors, min=0.0, max=self.max_prop_speed)  # rad/s
            self.previous_errors = errors

            return motors  # rad/s