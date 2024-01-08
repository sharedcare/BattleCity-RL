import random
from typing import SupportsFloat, Tuple, Dict, Any

import numpy as np
import retro
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.core import WrapperObsType
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._obs_buffer = np.zeros((2, *self.env.observation_space.shape),
                                    dtype=self.env.observation_space.dtype)
        obs, info = self.env.reset(seed, options)
        self._obs_buffer[0] = obs
        return obs, info


class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        boundary = 8
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        # crop frame to remote right panel
        frame = frame[boundary:-boundary, boundary:224-boundary, 0]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


if __name__ == "__main__":
    """
    Follow the instructions in https://github.com/akshysatish/DQN-BattleCityNes 
    to install BattleCity Gym Env
    """
    env = retro.make(game="BattleCity-Nes", players=1, state="stage1.state", record=".", render_mode="rgb_array")
    for i in range(1000):
        state = env.reset()
        rewards = 0
        while True:
            plt.figure(3)
            plt.clf()
            plt.imshow(env.render())
            plt.title("Reward: " + str(round(rewards, 2)))
            plt.axis('off')
            plt.show()
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward
            state = next_state
            if done:
                break
        print(rewards)
