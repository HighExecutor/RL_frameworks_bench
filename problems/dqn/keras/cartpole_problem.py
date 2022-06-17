import gym
import matplotlib.pyplot as plt
# from algs.dqn.keras_dqn_agent import KerasDqnAgent
# from algs.ddqn.keras_ddqn_agent import KerasDdqnAgent
# from algs.dueldqn.keras_dueldqn_agent import KerasDuelDqnAgent
from algs.dueldqn.keras_duelddqn_agent import KerasDuelDdqnAgent
from problems.problem import learn_problem, result_learning

def main():
    env = gym.make('CartPole-v1')
    state_space = 4
    action_space = 2
    layers = [24, 24, 24]
    agent = KerasDuelDdqnAgent(state_space, action_space, layers)
    need_render = True
    episodes = 200
    max_steps = 200
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

if __name__ == '__main__':
    main()

