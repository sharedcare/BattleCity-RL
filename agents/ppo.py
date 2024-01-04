import gym
import torch
import retro
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    GymWrapper,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm


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

    retro_env = retro.make(game="BattleCity-Nes", players=2, state="Start.2P.state", record=".", render_mode="rgb_array")
    env = GymWrapper(retro_env)

    env = TransformedEnv(
        env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(
                in_keys=["observation"],
            ),
            StepCounter(),
        ),
    )

    rollout = env.rollout(3)

