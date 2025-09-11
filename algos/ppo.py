import torch
import torch.nn as nn
from colorama import Fore, Style
import gymnasium as gym
import os

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 network_features=[256, 128]): 
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        layers = []
        in_channels = self.num_observations
        for features in network_features:
            layers.append(nn.Linear(in_channels, features))
            layers.append(nn.ELU())
            in_channels = features
        layers.append(nn.Linear(in_channels, self.num_actions))
        self.net = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        return mean, self.log_std_parameter, {}



class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 network_features=[256, 128]):
        Model.__init__(self, observation_space, action_space=1, device=device)
        DeterministicMixin.__init__(self)

        layers = []
        in_channels = self.num_observations
        for features in network_features:
            layers.append(nn.Linear(in_channels, features))
            layers.append(nn.ELU())
            in_channels = features
        layers.append(nn.Linear(in_channels, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class TrackAgent:
    def __init__(self,
                 env,
                 device="cuda",
                 base_cfg=None,
                 experiment_dir="runs/torch/LiftoffEnv"):
        
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs if hasattr(env, "num_envs") else 1
        self.experiment_dir = experiment_dir

        self.cfg = self._configure_agent(base_cfg)
        self.models = self._create_models()
        self.memory = self._create_memory()

        self.agent = PPO(models=self.models,
                         memory=self.memory,
                         cfg=self.cfg,
                         observation_space=self.observation_space,
                         action_space=self.action_space,
                         device=self.device)

    def _configure_agent(self, base_cfg):
        print(Fore.CYAN + "[INFO] Configuring PPO Agent..." + Style.RESET_ALL)
        cfg = PPO_DEFAULT_CONFIG.copy()
        if base_cfg is not None:
            cfg.update(base_cfg)

        cfg["rollouts"] = 256
        cfg["learning_epochs"] = 5
        cfg["mini_batches"] = 6
        cfg["discount_factor"] = 0.99
        cfg["lambda"] = 0.95
        cfg["learning_rate"] = 3e-4
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        cfg["random_timesteps"] = 0
        cfg["learning_starts"] = 0
        cfg["grad_norm_clip"] = 1.0
        cfg["ratio_clip"] = 0.2
        cfg["value_clip"] = 0.2
        cfg["clip_predicted_values"] = True
        cfg["entropy_loss_scale"] = 0.001
        cfg["value_loss_scale"] = 1.0
        cfg["kl_threshold"] = 0


        if isinstance(self.observation_space, gym.spaces.Box):
            cfg["state_preprocessor"] = RunningStandardScaler
            cfg["state_preprocessor_kwargs"] = {"size": self.observation_space, "device": self.device}
        else:
            print(Fore.YELLOW + "[WARN] Observation space is not Box. Disabling state preprocessor." + Style.RESET_ALL)
            cfg["state_preprocessor"] = None

        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}

        cfg["experiment"]["directory"] = self.experiment_dir
        cfg["experiment"]["write_interval"] = 180
        cfg["experiment"]["checkpoint_interval"] = 1800

        print(Fore.GREEN + "[INFO] PPO Agent Configured." + Style.RESET_ALL)
        return cfg

    def _create_models(self):
        print(Fore.CYAN + "[INFO] Creating Agent Models..." + Style.RESET_ALL)
        policy = Policy(self.observation_space, self.action_space, self.device)
        value = Value(self.observation_space, self.action_space, self.device)

        for model in [policy, value]:
            self._init_model_parameters(model)

        print(Fore.GREEN + "[INFO] Models Created." + Style.RESET_ALL)
        return {"policy": policy, "value": value}

    def _create_memory(self):
        print(Fore.CYAN + "[INFO] Creating Memory..." + Style.RESET_ALL)
        # The memory size should match the number of rollouts
        memory = RandomMemory(memory_size=self.cfg["rollouts"],
                            num_envs=self.num_envs,
                            device=self.device,
                            replacement=False)
        print(Fore.GREEN + "[INFO] Memory Created." + Style.RESET_ALL)
        return memory

    def _init_model_parameters(self, model):
        for name, param in model.named_parameters():
            try:
                if param.dim() > 1:
                    torch.nn.init.orthogonal_(param, gain=1.414)
                elif "bias" in name or "log_std" in name:
                    torch.nn.init.zeros_(param)
            except Exception as e:
                print(Fore.RED + f"  ERROR initializing {name}: {e}" + Style.RESET_ALL)

    def get_agent(self):
        """Returns the instantiated skrl PPO agent."""
        return self.agent    
            
