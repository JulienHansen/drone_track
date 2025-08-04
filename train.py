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
parser.add_argument('--num_envs', type=int, default=1000, help='Number of parallel environments')
parser.add_argument("--task", type=str, default="DefenseMARL", help="Name of the task.")
parser.add_argument("--total_timesteps", type=int, default=100000, help="Total simulation steps for training.")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="Timesteps between saving checkpoints.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)." )
parser.add_argument("--log_base_dir", type=str, default="runs", help="Base directory for logs and checkpoints.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Interval between video recordings (in steps).")
parser.add_argument("--algo", type=str, default="ippo", choices=["ippo", "p-ippo"], help="Select training algorithm: 'ippo' or 'p-ippo'.")
parser.add_argument("--population_size", type=int, default=1000, help="Max number of past attacker checkpoints to keep.")
parser.add_argument("--live_prob", type=float, default=0.4, help="Probability of using the live attacker vs. a frozen one.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from skrl.utils import set_seed
from env import DefenseEnv, DefenseEnvCfg
from algo import DefenseIPPOAgent as StandardAgentSetupClass

def main():
    global simulation_app
    device_id = app_launcher.device_id
    device = f"cuda:{device_id}"

    print(Fore.BLUE + f"[INFO] Using device: {device}" + Style.RESET_ALL)

    if args_cli.seed is not None:
        print(Fore.CYAN + f"[INFO] Setting random seed to: {args_cli.seed}" + Style.RESET_ALL)
        set_seed(args_cli.seed)
        random.seed(args_cli.seed)

    env_cfg = DefenseEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.headless:
        env_cfg.sim.render_interval = 1

    if args_cli.task not in gym.registry:
        gym.register(
            id=args_cli.task,
            entry_point="env:DefenseEnv",
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

    # Create pool directories for checkpoints
    pool_def_dir = os.path.join(log_dir, "pool_def")
    pool_atk_dir = os.path.join(log_dir, "pool_atk")
    os.makedirs(pool_def_dir, exist_ok=True)
    os.makedirs(pool_atk_dir, exist_ok=True)
    print(Fore.CYAN + f"[INFO] Defender checkpoints directory: {pool_def_dir}" + Style.RESET_ALL)
    print(Fore.CYAN + f"[INFO] Attacker checkpoints directory: {pool_atk_dir}" + Style.RESET_ALL)


    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0 and step > 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"{args_cli.algo}_step"
        }
        try:
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            print(Fore.YELLOW + f"[INFO] Recording videos to {video_folder}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Video init failed: {e}. Continuing without video." + Style.RESET_ALL)
            args_cli.video = False

    try:
        env = wrap_env(env, wrapper="isaaclab")
        print(Fore.GREEN + "[INFO] Environment wrapped successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to wrap env: {e}" + Style.RESET_ALL)
        traceback.print_exc()
        if simulation_app:
            simulation_app.close()
        return

    print(Fore.BLUE + "[INFO] Using Standard IPPO Agent Setup." + Style.RESET_ALL)
    AgentSetupClass = StandardAgentSetupClass
    agent_setup = AgentSetupClass(env=env, device=device, experiment_dir=log_dir)
    agent = agent_setup.get_agent()
    policy_models = {agent_id: agent.models[agent_id]["policy"] for agent_id in agent.possible_agents}

    atk_policy = policy_models["drone_attack"]
    def_policy = policy_models["drone_defense"]
    defender_agent_id = "drone_defense"
    attacker_agent_id = "drone_attack"

    total_timesteps = args_cli.total_timesteps
    steps_done = 0

    if args_cli.algo == "ippo":
        print(Fore.GREEN + f"[INFO] Running classic IPPO for {total_timesteps} timesteps." + Style.RESET_ALL)
        while steps_done < total_timesteps:
            chunk = min(args_cli.checkpoint_interval, total_timesteps - steps_done)
            cfg_train = {"timesteps": chunk, "headless": args_cli.headless}
            trainer = SequentialTrainer(cfg=cfg_train, env=env, agents=agent)
            trainer.train()
            steps_done += chunk
            # Save attacker and defender to their respective pool directories
            atk_ckpt = os.path.join(pool_atk_dir, f"attacker_{steps_done}.pt")
            def_ckpt = os.path.join(pool_def_dir, f"defender_{steps_done}.pt")
            torch.save(atk_policy.state_dict(), atk_ckpt)
            torch.save(def_policy.state_dict(), def_ckpt)
            print(Fore.GREEN + f"[INFO] Saved attacker checkpoint: {os.path.basename(atk_ckpt)} to {pool_atk_dir}" + Style.RESET_ALL)
            print(Fore.GREEN + f"[INFO] Saved defender checkpoint: {os.path.basename(def_ckpt)} to {pool_def_dir}" + Style.RESET_ALL)


    elif args_cli.algo == "p-ippo":
        print(
            Fore.GREEN
            + f"[INFO] Running P-IPPO (20% frozen-attacker, 80% co-train) for {total_timesteps} timesteps."
            + Style.RESET_ALL
        )

        past_attacker_checkpoints = deque(maxlen=args_cli.population_size)
        ckpt0 = os.path.join(pool_atk_dir, "attacker_0.pt")
        torch.save(atk_policy.state_dict(), ckpt0)
        past_attacker_checkpoints.append(ckpt0)
        latest_attacker_ckpt = ckpt0
        print(
            Fore.GREEN
            + f"[INFO] Saved initial attacker checkpoint: {os.path.basename(ckpt0)} to {pool_atk_dir}"
            + Style.RESET_ALL
        )
        for p in def_policy.parameters():
            p.requires_grad_(True)

        steps_done = 0
        while steps_done < total_timesteps:

            chunk = min(args_cli.checkpoint_interval, total_timesteps - steps_done)

            if random.random() < 0.4:

                for p in atk_policy.parameters():
                    p.requires_grad_(False)


                chosen_ckpt = random.choice(list(past_attacker_checkpoints))
                print(
                    Fore.YELLOW
                    + f"[INFO] (20%) Using frozen attacker: {os.path.basename(chosen_ckpt)}"
                    + Style.RESET_ALL
                )


                live_attacker_state = atk_policy.state_dict()
                atk_policy.load_state_dict(torch.load(chosen_ckpt, map_location=device))
  
                print(Fore.CYAN + f"[INFO] Training defender for {chunk} steps (frozen attacker)..." + Style.RESET_ALL)
                cfg_def = {
                    "timesteps": chunk,
                    "headless": args_cli.headless,
                    "agents_scope": [defender_agent_id],
                }
                trainer = SequentialTrainer(cfg=cfg_def, env=env, agents=agent)
                trainer.train()

                # Restore the live attacker weights back into atk_policy
                atk_policy.load_state_dict(live_attacker_state)
                for p in atk_policy.parameters():
                    p.requires_grad_(False)
                agent.models[attacker_agent_id]["policy"] = atk_policy

                def_ckpt = os.path.join(pool_def_dir, f"defender_{steps_done + chunk}.pt")
                torch.save(def_policy.state_dict(), def_ckpt)
                print(
                    Fore.GREEN
                    + f"[INFO] Saved defender checkpoint: {os.path.basename(def_ckpt)} to {pool_def_dir}"
                    + Style.RESET_ALL
                )

            else:
                # Unfreeze attacker so it can update; defender is already unfrozen
                for p in atk_policy.parameters():
                    p.requires_grad_(True)

                print(Fore.CYAN + f"[INFO] Co-training attacker & defender for {chunk} steps..." + Style.RESET_ALL)

                # Train both simultaneously 
                cfg_co = {
                    "timesteps": chunk,
                    "headless": args_cli.headless,
                    "agents_scope": [attacker_agent_id, defender_agent_id],
                }
                trainer = SequentialTrainer(cfg=cfg_co, env=env, agents=agent)
                trainer.train()
                live_ckpt = os.path.join(pool_atk_dir, f"attacker_{steps_done + chunk}.pt")
                torch.save(atk_policy.state_dict(), live_ckpt)
                past_attacker_checkpoints.append(live_ckpt)
                latest_attacker_ckpt = live_ckpt
                print(
                    Fore.GREEN
                    + f"[INFO] Saved co-trained attacker checkpoint: {os.path.basename(live_ckpt)} to {pool_atk_dir}"
                    + Style.RESET_ALL
                )
                def_ckpt = os.path.join(pool_def_dir, f"defender_{steps_done + chunk}.pt")
                torch.save(def_policy.state_dict(), def_ckpt)
                print(
                    Fore.GREEN
                    + f"[INFO] Saved co-trained defender checkpoint: {os.path.basename(def_ckpt)} to {pool_def_dir}"
                    + Style.RESET_ALL
                )

                for p in atk_policy.parameters():
                    p.requires_grad_(False)
                agent.models[attacker_agent_id]["policy"] = atk_policy
            steps_done += chunk

    if steps_done > 0:
        print(Fore.MAGENTA + "[INFO] Saving final checkpoints..." + Style.RESET_ALL)
        # Save final checkpoints to their respective pool directories
        final_def_ckpt = os.path.join(pool_def_dir, "defender_final.pt")
        torch.save(def_policy.state_dict(), final_def_ckpt)
        final_atk_ckpt = os.path.join(pool_atk_dir, "attacker_final.pt")
        torch.save(atk_policy.state_dict(), final_atk_ckpt)
        print(Fore.GREEN + f"[INFO] Saved final defender -> {final_def_ckpt}" + Style.RESET_ALL)
        print(Fore.GREEN + f"[INFO] Saved final attacker -> {final_atk_ckpt}" + Style.RESET_ALL)

    print(Fore.BLUE + "[INFO] Training finished. Closing environment." + Style.RESET_ALL)
    env.close()
    if simulation_app:
        print("[INFO] Closing Isaac Sim application...")
        simulation_app.close()
        print("[INFO] Isaac Sim application closed.")

if __name__ == "__main__":
    main()
