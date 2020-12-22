import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

#file = os.path.basename(__file__)
file = "compute_leaderboard.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/single"

seed = 0

ENV = "fishing-v1"    
env = gym.make(ENV, sigma = 0.1)

## TD3 ######################################################################


# [I 2020-11-24 18:03:25,390] Trial 29 finished with value: 7.736424446105957 and parameters: 
# hyper = {'gamma': 0.99, 'lr': 3.550974876596194e-05, 'batch_size': 64, 
# 'buffer_size': 100000, 'episodic': False, 'train_freq': 128,
# 'noise_type': 'ornstein-uhlenbeck', 'noise_std': 0.6615816256241758, 'net_arch': 'small'}

#Trial 15 finished with value: 7.6994757652282715 and parameters: 
hyper = {'gamma': 0.995, 'lr': 0.0001355522450968401, 'batch_size': 128, 
         'buffer_size': 10000, 'episodic': False, 'train_freq': 128, 
         'noise_type': 'normal', 'noise_std': 0.6656948079225263, 
         'net_arch': 'big'}

## current best from sigma0 tuning, not great    
#hyper= {
#    "batch_size": 32,
#    "buffer_size": 10000,
#    "episodic": False,
#    "gamma": 0.9999,
#    "lr": 0.000512517020156837,
#    "net_arch": "medium",
#    "noise_std": 0.9887668728863925,
#    "noise_type": "ornstein-uhlenbeck",
#    "train_freq": 2000 }


if hyper["net_arch"] == "big":
    policy_kwargs = dict(net_arch=[400, 300]) # big
else:
    policy_kwargs = dict(net_arch=[256, 256]) # medium


if hyper['episodic']:
    hyper['n_episodes_rollout'] = 1
    hyper['train_freq'], hyper['gradient_steps'] = -1, -1
else:
    hyper['train_freq'] = hyper['train_freq']
    hyper['gradient_steps'] = hyper['train_freq']
    hyper['n_episodes_rollout'] = -1


n_actions = env.action_space.shape[0]
if hyper["noise_type"] == "normal":
    hyper["action_noise"] = NormalActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )
elif hyper["noise_type"] == "ornstein-uhlenbeck":
    hyper["action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )

model = TD3('MlpPolicy', env,  verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            action_noise = hyper['action_noise'],
            train_freq = hyper['train_freq'],
            gradient_steps = hyper['train_freq'],
            n_episodes_rollout = hyper['n_episodes_rollout'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("TD3", ENV, mean_reward, std_reward, url)
print("algo:", "TD3", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3-r01.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/td3-policy-r01.png")
model.save("models/td3-r01")
