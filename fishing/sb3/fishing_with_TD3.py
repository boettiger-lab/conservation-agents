import gym
import gym_fishing
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard
import os

ENV = "fishing-v1"

env = gym.make(ENV)
model = TD3('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=200000)
#model.load("models/td3")

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("TD3", ENV, mean_reward, std_reward, os.path.basename(__file__))


model.save("models/td3")


