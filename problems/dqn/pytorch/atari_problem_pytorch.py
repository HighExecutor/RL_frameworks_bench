import gym
import matplotlib.pyplot as plt
from algs.dqn.pytorch_dqn_agent import PytorchDqnAgent
from problems.problem import learn_problem

def main():
    env = gym.make('DemonAttack-ram-v0')
    state_space = 128
    action_space = 6
    layers = [256, 256, 32]
    agent = PytorchDqnAgent(state_space, action_space, layers, memory_size=100000, batch_size=32)
    need_render = False
    episodes = 2000
    max_steps = 10000
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':
    main()