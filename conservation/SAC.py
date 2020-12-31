from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, sac

import gym
import gym_fishing

env_id = "conservation-v5"
algo="sac"
outdir = "results"
total_timesteps = 300000
verbose = 0
seed = 0
tensorboard_log = None


# Override defaults
use_sde = True

env = gym.make(env_id)
model = sac(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = None, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = SAC('MlpPolicy', env, verbose = 0, tensorboard_log="/var/log/tensorboard/single", 
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")
