from isaaclab.app import AppLauncher
import argparse
import random
import torch
import gymnasium as gym
import os
import datetime
import traceback
from colorama import Fore, Style
from collections import deque
import copy

parser = argparse.ArgumentParser(description='DefenseEnv MARL Training with IPPO Variants')
parser.add_argument('--num_envs', type=int, default=2000, help='Number of parallel environments')
parser.add_argument("--task", type=str, default="DefenseMARL", help="Name of the task.")
parser.add_argument("--total_timesteps", type=int, default=100000, help="Total simulation steps for training.")
parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Timesteps between saving checkpoints.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)." )
parser.add_argument("--log_base_dir", type=str, default="runs", help="Base directory for logs and checkpoints.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")
parser.add_argument("--algo", type=str, default="ippo", choices=["ippo", "p-ippo"], help="Select training algorithm: 'ippo' or 'p-ippo'.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from tasks.eight_track import EightEnv, EightEnvCfg
from algos.ppo import TrackAgent


def main():
    global simulation_app
    device_id = app_launcher.device_id
    device = f"cuda:{device_id}"
    if args_cli.seed is not None:
        print(Fore.CYAN + f"[INFO] Setting random seed to: {args_cli.seed}" + Style.RESET_ALL)
        set_seed(args_cli.seed)
        random.seed(args_cli.seed)

    # Initialize environment
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
    env = gym.make(args_cli.task, render_mode=render_mode)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args_cli.task}_{args_cli.algo.upper()}_{timestamp}"
    log_dir = os.path.abspath(os.path.join(args_cli.log_base_dir, experiment_name))
    os.makedirs(log_dir, exist_ok=True)
    print(Fore.MAGENTA + f"[INFO] Log directory: {log_dir}" + Style.RESET_ALL)

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0 and step > 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"{args_cli.algo}_step"
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        args_cli.video = False  

    env = wrap_env(env, wrapper="isaaclab")
    agent_setup = TrackAgent(env=env, device=device, experiment_dir=log_dir)
    agent = agent_setup.get_agent()


    trainer = SequentialTrainer(
        cfg={
            "timesteps": args_cli.total_timesteps,
            "headless": args_cli.headless,
            "checkpoint_interval": args_cli.checkpoint_interval,
            "log_dir": log_dir
        },
        env=env,
        agents=agent,
    )
    
    print(Fore.YELLOW + "[INFO] Starting training..." + Style.RESET_ALL)
    trainer.train()
    model_path = os.path.join(log_dir, "policy_model.pt")
    torch.save(agent.models["policy"].state_dict(), model_path)

    print(Fore.GREEN + f"[INFO] Model saved to {model_path}" + Style.RESET_ALL)
    simulation_app.close()

if __name__ == "__main__":
    main()
