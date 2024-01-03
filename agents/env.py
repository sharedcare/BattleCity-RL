import retro
import gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Follow the instructions in https://github.com/akshysatish/DQN-BattleCityNes 
    to install BattleCity Gym Env
    """
    env = retro.make(game="BattleCity-Nes", players=2, state="Start.2P.state", record=".")
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode="rgb_array"))
    plt.title("Game Frame")
    plt.axis('off')
    plt.show()
