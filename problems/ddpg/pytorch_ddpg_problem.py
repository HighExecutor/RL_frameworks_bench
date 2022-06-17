import gym
import matplotlib.pyplot as plt
from problems.problem import result_learning
from algs.ddpg.ddpg_pytorch_agent import Agent


def main():
    env = gym.make("LunarLanderContinuous-v2")
    state_space = 8
    action_space = 2
    layers = [400, 300]
    agent = Agent(input_dims=state_space, n_actions=action_space, gamma=0.99, max_size=100000, layer1_size=layers[0],
                  layer2_size=layers[1], batch_size=64, tau=0.001)
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
