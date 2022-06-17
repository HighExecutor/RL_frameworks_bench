import gym
import matplotlib.pyplot as plt
from problems.problem import result_learning
from algs.ddpg.ddpg_keras_agent import KerasDdpgAgent


def main():
    env = gym.make("LunarLanderContinuous-v2")
    state_space = 8
    action_space = 2
    layers = [200, 120]
    agent = KerasDdpgAgent(state_size=state_space, action_size=action_space,
                           action_min=env.action_space.low[0], action_max=env.action_space.high[0],
                           layers=layers, lr1=0.00001, lr2=0.00001, gamma=0.99, tau=0.01,
                           mem_size=100000, batch_size=64)
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
