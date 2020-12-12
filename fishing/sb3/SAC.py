import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

#file = os.path.basename(__file__)
file = "compute_leaderboard.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/leaderboard"

seed = 0

ENV = "fishing-v1"    
env = gym.make(ENV)
# FIXME read from YAML
hyper = {'gamma': 0.99, 
         'lr': 1.8825727360507924e-05, 
         'batch_size': 512, 
         'buffer_size': 10000, 
         'learning_starts': 10000, 
         'train_freq': 1, 
         'tau': 0.005, 
         'log_std_init': -0.3072998266889968, 
         'net_arch': 'medium'}
         
policy_kwargs = dict(log_std_init=hyper["log_std_init"], net_arch=[256, 256])
model = SAC('MlpPolicy', 
            env, verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            use_sde=True,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            learning_starts = hyper['learning_starts'],
            train_freq = hyper['train_freq'],
            tau = hyper['tau'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000)

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("SAC", ENV, mean_reward, std_reward, url)
print("algo:", "SAC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/sac-policy.png")

model.save("models/sac-v1")
