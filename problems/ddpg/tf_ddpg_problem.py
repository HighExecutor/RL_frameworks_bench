import gym
import matplotlib.pyplot as plt
from problems.problem import result_learning
from algs.ddpg.ddpg_tf_agent import Agent


def main():
    env = gym.make("LunarLanderContinuous-v2")
    state_space = 8
    action_space = 2
    layers = [200, 120]
    agent = Agent(alpha=0.0005, beta=0.0005, input_dims=8, tau=0.01,
                  env=env, batch_size=64, layer1_size=200, layer2_size=120,
                  n_actions=2)
    need_render = True
    episodes = 2000
    max_steps = 500
    scores = learn(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)


def learn(env, agent, episodes, max_steps, need_render):
    scores = []
    for e in range(episodes):
        state = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.learn()
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores


if __name__ == "__main__":
    main()
