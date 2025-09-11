import torch
import matplotlib.pyplot as plt
from drone_dynamics import *
from PIDController import *


# --- Paramètres ---
device = "cpu"
num_envs = 1
dt = 0.01   # 100 Hz
steps = 200  # ~4 secondes

# --- Init dynamique ---
dyn = DroneDynamics(num_envs=num_envs, device=device)
dyn.reset(env_ids=torch.arange(num_envs))

# --- Etats initiaux ---
omega_b = torch.zeros(num_envs, 3)  # body rates [p,q,r]
thrust_measured = torch.zeros(num_envs, 1)  # thrust

# --- Consignes (setpoint) ---
target_thrust = torch.tensor([[10.0]], device=device)  
target_rates = torch.tensor([[1.0, 2.0, 3.0]], device=device)  
setpoint = torch.cat([target_thrust, target_rates], dim=1)  # [N,4] -> thrust, roll, pitch, yaw

# --- Init PID ---
pid = PIDController(num_envs=num_envs, device=device, setpoint=setpoint)

# Logs
log_time, log_rates, log_rates_target, log_thrust, log_thrust_target = [], [], [], [], []

# --- Boucle simulation ---
for step in range(steps):
    t = step * dt

    # --- Process variable : thrust + body rates ---
    process_variable = torch.cat([thrust_measured, omega_b], dim=1)  # [N,4]

    # --- PID → commandes moteurs ---
    motors = pid.compute(process_variable, dt=dt)

    # --- Dynamiques forces/torques ---
    forces_b, torques_b = dyn.compute_forces_and_torques(motors, lin_vel_b=torch.zeros(num_envs,3), dt=dt)

    # --- Update body rates ---
    omega_b = dyn.step_body_rates(torques_b, omega_b, dt)

    # --- Mesure thrust (composante z) ---
    thrust_measured = forces_b[:, 2].unsqueeze(1)  # [N,1]

    # --- Logs ---
    log_time.append(t)
    log_rates.append(omega_b[0].detach().numpy())
    log_rates_target.append(target_rates[0].numpy())
    log_thrust.append(thrust_measured[0,0].item())
    log_thrust_target.append(target_thrust[0,0].item())

# --- Convert logs ---
import numpy as np
log_time = np.array(log_time)
log_rates = np.array(log_rates)
log_rates_target = np.array(log_rates_target)
log_thrust = np.array(log_thrust)
log_thrust_target = np.array(log_thrust_target)

# --- Plots ---
plt.figure(figsize=(10,6))

# Body rates
plt.subplot(2,1,1)
line_roll, = plt.plot(log_time, log_rates[:,0], label="Roll rate (p)")
line_pitch, = plt.plot(log_time, log_rates[:,1], label="Pitch rate (q)")
line_yaw, = plt.plot(log_time, log_rates[:,2], label="Yaw rate (r)")
plt.plot(log_time, log_rates_target[:,0], linestyle='--', color=line_roll.get_color(), label="Roll Target")
plt.plot(log_time, log_rates_target[:,1], linestyle='--', color=line_pitch.get_color(), label="Pitch target")
plt.plot(log_time, log_rates_target[:,2], linestyle='--', color=line_yaw.get_color(), label="Yaw target")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Body rates [rad/s]")

# Thrust
plt.subplot(2,1,2)
plt.plot(log_time, log_thrust, label="Thrust measured")
plt.plot(log_time, log_thrust_target, "k--", label="Thrust target")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")

plt.tight_layout()
plt.show()
