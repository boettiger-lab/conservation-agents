from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import a2c

import numpy as np
import os
import shutil

import gym
import gym_fishing

env_id = "fishing-v1"
algo="a2c"
outdir = "results"
total_timesteps = 300000
seed = 0

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- A2C defaults
default_a2c = {
"params_activation_fn": "tanh",
"params_ent_coef": 0.0,
"params_gae_lambda": 1.0,
"params_gamma": 0.99,
"params_log_std_init": 0.0,
"params_lr": 0.0007,
"params_lr_schedule": "linear",
"params_max_grad_norm": 0.5,
"params_n_steps": 5,
"params_net_arch": "small",
"params_normalize_advantage": False,
"params_ortho_init": True,
"params_use_rms_prop": True,
"params_vf_coef":  0.5
}

env = make_vec_env(env_id, n_envs=4, seed = seed)
model = a2c(env, default_a2c, "MlpPolicy", seed = seed)
model.learn(total_timesteps = total_timesteps)

# eval env
env = Monitor(gym.make(env_id))


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("env_id ", "mean reward:", mean_reward, "std:", std_reward)


## ensure a path to work on
dest = os.path.join(outdir, env_id, algo)
if os.path.exists(dest):
  shutil.rmtree(dest)
os.makedirs(dest)
model.save(os.path.join(dest, "agent"))

## simulate and plot results
np.random.seed(seed)
df = env.simulate(model, reps=10)
df.to_csv(os.path.join(dest, "sim.csv"))
env.plot(df, os.path.join(dest, "sim.png"))
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, os.path.join(dest, "policy.png"))
