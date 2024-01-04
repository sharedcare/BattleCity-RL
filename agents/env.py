import random

import numpy as np
import retro
import gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Follow the instructions in https://github.com/akshysatish/DQN-BattleCityNes 
    to install BattleCity Gym Env
    """
    env = retro.make(game="BattleCity-Nes", players=2, state="Start.2P.state", record=".", render_mode="rgb_array")
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
            rewards += reward[0]
            state = next_state
            if done:
                break
        print(rewards)
