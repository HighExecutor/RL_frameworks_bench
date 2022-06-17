import gym
import matplotlib.pyplot as plt
# from algs.dqn.keras_dqn_agent import KerasDqnAgent
from algs.dueldqn.keras_duelddqn_agent import KerasDuelDdqnAgent
import numpy as np

def main():
    env = gym.make('SpaceInvaders-ram-v0')
    state_space = 128
    action_space = 6
    layers = [64, 64, 16]
    agent = KerasDuelDdqnAgent(state_space, action_space, layers, memory_size=100000, batch_size=256)
    need_render = True
    episodes = 2000
    max_steps = 50000
    scores = learn_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()

def learn_problem(env, agent, episodes, max_steps, need_render):
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
            agent.remember(state, action, reward, next_state, int(done))
            state = next_state
            agent.replay()
            if done:
                break
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores

if __name__ == '__main__':
    main()