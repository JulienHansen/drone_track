import argparse
import os
import time

import gymnasium as gym
import torch
from colorama import Fore, Style

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of a custom agent.")
parser.add_argument("--run_dir", type=str, required=True, help="Path to the experiment directory created by train.py.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="DefenseMARL", help="Name of the task (must match training).")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the gameplay.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from skrl.envs.wrappers.torch import wrap_env
from tasks.eight_track import EightEnv, EightEnvCfg
from algos.ppo import TrackAgent



def main():
    global simulation_app
    device_id = app_launcher.device_id
    device = f"cuda:{device_id}"
    print(f"[INFO] Loading experiment from directory: {args_cli.run_dir}")

    env_cfg = EightEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.headless:
        env_cfg.sim.render_interval = 1 

    if args_cli.task not in gym.registry:
        gym.register(
            id=args_cli.task,
            entry_point="tasks.eight_track:EightEnv",
            kwargs={'cfg': env_cfg},
            disable_env_checker=True
        )

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    
    try:
        dt = env.unwrapped.dt
    except AttributeError:
        dt = env.unwrapped.step_dt
        
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.run_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "gameplay_best_agent"
        }
        print(Fore.CYAN + "[INFO] Recording video..." + Style.RESET_ALL)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = wrap_env(env, wrapper="isaaclab")

    agent_setup = TrackAgent(env=env, device=device, experiment_dir=args_cli.run_dir)
    agent = agent_setup.get_agent()

    try:
        skrl_subdirs = [d for d in os.listdir(args_cli.run_dir) if os.path.isdir(os.path.join(args_cli.run_dir, d)) and d != "videos"]
        if len(skrl_subdirs) != 1:
            raise FileNotFoundError(f"Expected 1 log subdirectory in '{args_cli.run_dir}', but found {len(skrl_subdirs)}: {skrl_subdirs}")
        skrl_subdir = skrl_subdirs[0]
        
        checkpoint_dir = os.path.join(args_cli.run_dir, skrl_subdir, "checkpoints")
        model_path = os.path.join(checkpoint_dir, "best_agent.pt")
        
        if not os.path.exists(model_path):
            print(Fore.YELLOW + f"[WARNING] 'best_agent.pt' not found. Trying to find the latest checkpoint." + Style.RESET_ALL)
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('agent_') and f.endswith('.pt')]
            if not checkpoints:
                raise FileNotFoundError("No 'best_agent.pt' or any 'agent_*.pt' checkpoints found.")
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)

    except FileNotFoundError as e:
        print(Fore.RED + f"[ERROR] Could not find checkpoint file: {e}" + Style.RESET_ALL)
        simulation_app.close()
        return

    print(Fore.GREEN + f"[INFO] Loading agent checkpoint from: {model_path}" + Style.RESET_ALL)
    agent.load(model_path) 
    
    agent.set_running_mode("eval")

    obs, _ = env.reset()
    timestep = 0
    
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = agent.act(obs, timestep=timestep, timesteps=timestep)[0]
            obs, rewards, terminated, truncated, info = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                print(Fore.CYAN + f"Finished recording video. Saved in {os.path.join(args_cli.run_dir, 'videos', 'play')}" + Style.RESET_ALL)
                break
        
        if args_cli.real_time and not args_cli.headless:
            elapsed_time = time.time() - start_time
            sleep_time = dt - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
    finally:
        if simulation_app:
            simulation_app.close()