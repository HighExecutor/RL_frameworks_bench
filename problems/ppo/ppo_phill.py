import gym
import numpy as np
from algs.ppo.ppo_phill_nomicrobatch import Agent
import matplotlib.pyplot as plt


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
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    max_steps = 500
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)
    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        env.render()
        done = False
        score = 0
        for j in range(max_steps):
            action, prob, val = agent.choose_action(observation)
            env.render()
            observation_, reward, done, info = env.step(action)
            # if j == max_steps - 1:
            #     reward -= 200.0
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn(4)
                learn_iters += 1
            observation = observation_
            if done:
                break
        # agent.learn(4)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]

    plt.plot(score_history)
    plt.show()

    result_learning(env, agent, max_steps)

