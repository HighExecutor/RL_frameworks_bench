import gym

from stable_baselines import DQN, A2C, DDPG, SAC, TD3, PPO1
from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines.deepq.policies import DQNPolicy, FeedForwardPolicy, MlpPolicy
from stable_baselines.common.policies import BasePolicy, ActorCriticPolicy
# from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines.td3.policies import MlpPolicy
from gym.wrappers import TimeLimit


# class CustomPolicy(BasePolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs,
#                                            layers=[200, 120])

# Create environment
# env = gym.make('SpaceInvaders-ram-v0')
# env = gym.make('LunarLander-v2')
if __name__ == "__main__":
    # env_vec = SubprocVecEnv([lambda: TimeLimit(gym.make('LunarLander-v2'), max_episode_steps=1000) for _ in range(3)])
    env_vec = gym.make('LunarLanderContinuous-v2')
    print("process_started")

# Instantiate the agent
    model = PPO2(ActorCriticPolicy, env_vec, verbose=1, policy_kwargs={"net_arch": [100, 80]})
# Train the agent
    episodes = 1000
    steps = episodes * 500
    learn_result = model.learn(total_timesteps=int(steps))
# Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# model = DQN.load("dqn_lunar")

# Evaluate the agent
    test_env = gym.make('LunarLanderContinuous-v2')
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print(mean_reward)
    print(std_reward)

    # Enjoy trained agent
    for _ in range(10000):
        obs = test_env.reset()
        score = 0.0
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = test_env.step(action)
            score += rewards
            test_env.render()
            if dones:
                print(score)
                break
