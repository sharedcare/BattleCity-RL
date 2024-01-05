import gym
import torch
import retro
import ray
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
from torch import nn


def random_rollout(env: gym.Env):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards += reward

    return rewards


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

    ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)

    env = retro.make(game="BattleCity-Nes", players=2, state="Start.2P.state", record=".",
                     render_mode="rgb_array")

    algo = ppo.PPO(env=env)
    while True:
        print(algo.train())
