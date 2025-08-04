import os
import torch
import torch.nn as nn 
import gymnasium as gym 

from colorama import Fore, Style
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions = True,
                 clip_log_std = True, min_log_std = 20, max_log_std = 2, reduction = "sum",
                 network_features = [256, 128]):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        layers = []
        in_channels = self.num_observations # Attributes come from Model 
        
        for features in network_features:
            layers.append(nn.Linear(in_channels, features))
            layers.append(nn.ELU())
            in_channels = features
        
        layers.append(nn.Linear(in_channels, self.num_actions))
        self.net = nn.Sequential(*layers) # nn.Sequential to dynamically chains the layers together
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        return mean, self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, network_features = [256, 128]):

        Model.__init__(self, observation_space, action_space = 1, device = device)
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
    def __init__(self, env, device = "cuda", base_cfg = None, experiment_dir = "runs/torch"):

        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs if hasattr(env, "num_envs") else 1
        self.experiment_dir = experiment_dir

        self.cfg = self._configure_agent(base_cfg)
        self.models = self._create_models()
        self.memory = self._create_memory()

        self.agent = PPO(models = self.models, 
                         memory = self.memory,
                         cfg = self.cfg,
                         observation_space = self.observation_space,
                         action_space = self.action_space,
                         device = self.device)

    def _configure_agent(self, base_cfg):
        cfg = PPO_DEFAULT_CONFIG.copy()
        
        if base-cfg is not None:
            cfg.update(base_cfg)

        cfg["rollouts"] = 24
        cfg["learning epochs"] = 5
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

        first_agent = self.possible_agents[0]
        obs_space = self.observation_space[first_agent]

        if isinstance(obs_space, gym.spaces.Box):
             cfg["state_preprocessor"] = RunningStandardScaler
             cfg["state_preprocessor_kwargs"] = {"size": obs_space, "device": self.device}
             self._state_preprocessor_enabled = True # 
        else:
             print(Fore.YELLOW + f"[WARN] Observation space for {first_agent} is not Box. Disabling state preprocessor." + Style.RESET_ALL)
             cfg["state_preprocessor"] = None
             self._state_preprocessor_enabled = False

        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}
        cfg["experiment"]["directory"] = self.experiment_dir
        cfg["experiment"]["write_interval"] = 180
        cfg["experiment"]["checkpoint_interval"] = 1800

        print(Fore.GREEN + "[INFO] IPPO Agent Configured." + Style.RESET_ALL)
        return cfg
    
    def _create_memory(self):
        memories = {}
        memory_size = self.cfg["rollouts"]

        for agent_id in self.possible_agents:
            memories[agent_id] = RandomMemory( memory_size = memory_size, num_envs = self.num_envs
                                               device = self.device, replacement = False)
        return memories

    def _create_models(self):

        models = {}
        for agent_id in self.possible_agents:
            models[agent_id] = {}
            models[agent_id]["policy"] = Policy(self.observation_space[agent_id],
                                                self.action_space[agent_id],
                                                self.device)
            models[agent_id]["value"] = Value(self.observation_space[agent_id],
                                              action_space = 1,
                                              device = self.device)

            for model in models[agent_id].values():
                self._init_model_parameters()

        return models

    def _init_model_parameters(self, model):
         for name, param in model.named_parameters():
             if param.dim() > 1:
                 try: torch.nn.init.orthogonal_(param, gain=1.414)
                 except Exception as e: print(Fore.RED + f"  ERROR initializing weights {name} with Orthogonal: {e}" + Style.RESET_ALL)
             elif "bias" in name:
                 try: torch.nn.init.zeros_(param)
                 except Exception as e: print(Fore.RED + f"  ERROR initializing bias {name} with Zeros: {e}" + Style.RESET_ALL)
             elif "log_std" in name:
                 try: torch.nn.init.zeros_(param)
                 except Exception as e: print(Fore.RED + f"  ERROR initializing {name} with Zeros: {e}" + Style.RESET_ALL)

    def get_agent(self):
        """Returns the instantiated skrl IPPO agent."""
        return self.agent                
            
