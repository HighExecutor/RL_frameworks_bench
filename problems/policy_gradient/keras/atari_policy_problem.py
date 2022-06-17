import gym
import matplotlib.pyplot as plt
from algs.vpg.keras_vpg_agent import KerasVpgAgent
from problems.problem import result_learning
import numpy as np

def main():
    env = gym.make('SpaceInvaders-ram-v0')
    state_space = 128
    action_space = 6
    layers = [512, 512, 64]
    agent = KerasVpgAgent(state_space, action_space, layers)
    need_render = True
    episodes = 5000
    max_steps = 20000
    scores = learn_policy_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

def learn_policy_problem(env, agent, episodes, max_steps, need_render):
    scores = []
    for e in range(episodes):
        state = env.reset()
        state = np.array(state) / 255.0
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state) / 255.0
            score += reward
            agent.remember(state, action, reward)
            state = next_state
            if done:
                break
        agent.learn()
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores

if __name__ == '__main__':
    main()

