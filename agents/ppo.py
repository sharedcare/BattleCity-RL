from typing import Optional, Tuple

import gymnasium as gym
import torch
import retro
import numpy as np
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv


# class RetroGymEnv(gym.Env):
#     def __init__(self, env_config):
#         self.env = retro.make(game=env_config["game"], players=env_config["num_players"], state=env_config["state"], record=".",
#                               render_mode=env_config["render_mode"])
#         self.observation_space = self.env.observation_space
#         self.action_space = gym.spaces.Discrete(self.env.action_space.n)
#
#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[dict] = None,
#     ) -> Tuple[ObsType, dict]:
#         return self.env.reset(seed, options)
#
#     def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
#         bin_action = np.binary_repr(action, width=self.action_space.n)
#         bin_action
#         return self.env.step(action)


if __name__ == "__main__":
    # Hyperparams
    device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
    hidden_dim = 256
    lr = 1e-4
    max_grad_norm = 1.0

    # PPO params
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimisation steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    # ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)

    env = retro.make(game="BattleCity-Nes", players=1, state="Start.2P.state", record="./logs",
                     render_mode="rgb_array")
    env = MaxAndSkipEnv(env)
    env = WarpFrame(env)

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)

    # register_env("BattleCity", RetroGymEnv)

    # ray.rllib.utils.check_env(RetroGymEnv(env_config={"game": "BattleCity-Nes",
    #                                                   "num_players": 1,
    #                                                   "state": "Start.2P.state",
    #                                                   "render_mode": "rgb_array",
    #                                                   }))

    # algo = ppo.PPO(env="BattleCity", config={
    #     "env_config": {"game": "BattleCity-Nes",
    #                    "num_players": 1,
    #                    "state": "Start.2P.state",
    #                    "render_mode": "rgb_array",
    #                    },
    # })
    # while True:
    #     print(algo.train())
