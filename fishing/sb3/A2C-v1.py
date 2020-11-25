import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import A2C, TD3, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

file = os.path.basename(__file__)
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/single"

ENV = "fishing-v1"    
env = gym.make(ENV)
model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log )
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v4.png")


ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2Cv1_1.png")


ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.3)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2Cv1_2.png")

ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.4)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2Cv1_3.png")


ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.1)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2Cv1_4.png")
