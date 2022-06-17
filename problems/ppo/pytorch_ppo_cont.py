import gym
import numpy as np
from algs.ppo.ppo_cont_pytorch_agent import Agent
import matplotlib.pyplot as plt

def main():
    env = gym.make('LunarLanderContinuous-v2')
    N = 2048
    batch_size = 64
    n_epochs = 8
    alpha = 0.0003
    state_space = env.observation_space.shape
    action_space = env.action_space.shape[0]
    layers = [150, 120]
    agent = Agent(n_actions=action_space, batch_size=batch_size, input_dims=state_space,
                  layers=layers, alpha=alpha, n_epochs=n_epochs)
    need_render = False
    episodes = 2000
    max_steps = 1000
    scores = learn_policy_problem(env, agent, episodes, max_steps, N, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

def learn_policy_problem(env, agent, episodes, max_steps, N, need_render):
    scores = []
    step = 0
    for e in range(episodes):
        state = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.store(state, action, prob, val, reward, done)
            if step % N == 0:
                agent.learn()
            step += 1
            state = next_state
            if done:
                break
        # agent.learn()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores

def result_learning(env, agent, max_steps):
    agent.epsilon = 0.0
    while True:
        state = env.reset()
        score = 0
        for i in range(max_steps):
            env.render()
            action, _, _ = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break
        print("score: {}".format(score))

if __name__ == '__main__':
    main()