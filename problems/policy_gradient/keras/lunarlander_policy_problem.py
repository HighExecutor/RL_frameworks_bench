import gym
import matplotlib.pyplot as plt
from algs.vpg.keras_vpg_agent import KerasVpgAgent
from problems.problem import learn_policy_problem, result_learning

def main():
    env = gym.make('LunarLander-v2')
    state_space = 8
    action_space = 4
    layers = [200, 120]
    agent = KerasVpgAgent(state_space, action_space, layers)
    need_render = False
    episodes = 5000
    max_steps = 1000
    scores = learn_policy_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

if __name__ == '__main__':
    main()

