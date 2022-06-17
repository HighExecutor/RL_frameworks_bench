import gym
import matplotlib.pyplot as plt
from algs.dqn.pytorch_dqn_agent import PytorchDqnAgent
from problems.problem import learn_problem, result_learning

def main():
    env = gym.make('CartPole-v1')
    state_space = 4
    action_space = 2
    layers = [24, 24, 24]
    agent = PytorchDqnAgent(state_space, action_space, layers)
    need_render = False
    episodes = 200
    max_steps = 200
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

if __name__ == '__main__':
    main()

