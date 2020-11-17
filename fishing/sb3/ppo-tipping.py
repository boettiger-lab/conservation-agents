import numpy as np
import gym
import gym_fishing
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import leaderboard
import os
url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v2" # A2C can do discrete & cts
env = gym.make(ENV, C = 0.2)
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="/var/log/tensorboard/vec", device = "cpu")
model.learn(total_timesteps=300000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo-tipping.png")


