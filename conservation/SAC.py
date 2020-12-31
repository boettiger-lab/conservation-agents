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
tensorboard_log="/var/log/tensorboard/single"

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- A2C defaults
hyper = {
"params_lr": 0.0003,
"params_gamma": 0.99,
"params_batch_size": 256,
"params_buffer_size": 1000000,
"params_learning_starts": 100,
"params_train_freq": 1,
"params_tau": 0.005,
"params_log_std_init": -3,
"params_net_arch": "small",
"value": 0 # only in logs
}

## defaults, not tuned
gradient_steps=1 # tuner sets equal to train_freq
n_episodes_rollout=- 1
action_noise=None
optimize_memory_usage=False
ent_coef='auto'
target_update_interval=1
target_entropy='auto'
use_sde=False
sde_sample_freq=- 1
use_sde_at_warmup=False
# Alternately, load tuned values  from log dir
#hyper = best_hyperpars(log_dir, env_id, algo)
# Override defaults
use_sde = True


env = gym.make(env_id)
model = sac(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = SAC('MlpPolicy', env, verbose = 0, tensorboard_log = tensorboard_log,
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")
