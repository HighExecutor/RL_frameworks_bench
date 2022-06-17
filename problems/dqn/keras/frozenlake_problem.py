import gym
import matplotlib.pyplot as plt
from algs.dqn.keras_dqn_agent import KerasDqnAgent
from problems.problem import learn_problem

def main():
    env = gym.make('FrozenLake-v0')
    state_space = 4
    action_space = 4
    layers = [32, 32]
    agent = KerasDqnAgent(state_space, action_space, layers)

    need_render = True
    episodes = 2000
    max_steps = 200
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':
    main()

