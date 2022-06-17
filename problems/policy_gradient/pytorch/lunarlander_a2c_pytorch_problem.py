import gym
import matplotlib.pyplot as plt
from algs.a2c.pytorch_a2c_agent import PytorchA2cAgent
from problems.problem import result_learning

def main():
    env = gym.make('LunarLander-v2')
    state_space = 8
    action_space = 4
    layers = [400, 200]
    agent = PytorchA2cAgent(state_space, action_space, layers)
    need_render = False
    episodes = 2000
    max_steps = 500
    scores = learn_policy_problem(env, agent, episodes, max_steps, need_render)
    plt.plot(scores)
    plt.show()
    result_learning(env, agent, max_steps)

def learn_policy_problem(env, agent, episodes, max_steps, need_render):
    scores = []
    for e in range(episodes):
        state = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action, _ = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.store_rewards(reward)
            state = next_state
            if done:
                break
        agent.learn(state)
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores

if __name__ == '__main__':
    main()

