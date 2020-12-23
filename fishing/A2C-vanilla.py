from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, tune_best, make_policy_kwargs

import numpy as np
import os
import shutil

import gym
import gym_fishing

env_id = "fishing-v1"
algo="a2c"
logs_dir = "logs"            
outdir = "vanilla"
total_timesteps = 300000
verbose = 1
seed = 0
tensorboard_log = None

## Configured in hyperparams/a2c.yml
use_sde = True
n_envs = 4

#env = gym.make(env_id)
env = make_vec_env(env_id, n_envs=n_envs, seed = seed)

model = A2C('MlpPolicy', 
            env, 
            verbose = verbose, 
            tensorboard_log=tensorboard_log, 
            seed = seed,
            use_sde = use_sde)

model.learn(total_timesteps = total_timesteps)

# eval env
env = Monitor(gym.make(env_id))

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("env_id ", "mean reward:", mean_reward, "std:", std_reward)

# ensure a path to work on
dest = os.path.join(outdir, env_id, algo)
if os.path.exists(dest):
  shutil.rmtree(dest)
os.makedirs(dest)
model.save(os.path.join(dest, "agent"))

# simulate and plot results
np.random.seed(seed)
df = env.simulate(model, reps=10)
df.to_csv(os.path.join(dest, "sim.csv"))
env.plot(df, os.path.join(dest, "sim.png"))
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, os.path.join(dest, "policy.png"))
