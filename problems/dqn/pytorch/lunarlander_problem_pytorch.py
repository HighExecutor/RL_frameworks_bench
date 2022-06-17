import gym
import matplotlib.pyplot as plt
from algs.dqn.pytorch_dqn_agent import PytorchDqnAgent
from problems.problem import learn_problem, result_learning

def main():
    env = gym.make('LunarLander-v2')
    state_space = 8
    action_space = 4
    layers = [200, 120]
    agent = PytorchDqnAgent(state_space, action_space, layers, batch_size=32, memory_size=20000)
    need_render = False
    episodes = 500
    max_steps = 1000
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

if __name__ == '__main__':
    main()

