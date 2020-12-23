from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, ddpg

import gym, gym_fishing

env_id = "fishing-v1"
algo = "ddpg"
outdir = "results"
total_timesteps = 100000
verbose = 0
seed = 0
tensorboard_log = None

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- defaults
hyper = {
"params_gamma": 0.99,
"params_tau": 0.005,
"params_lr": 0.001,
"params_batch_size": 100,
"params_buffer_size": 1000000,
"params_noise_std": 0.1, # sets action noise size -- cannot find default size!
"params_episodic": True,  # this actually sets the next three
#"params_train_freq": -1,
#"params_gradient_steps": -1,
#"params_n_episodes_rollout": 1,
"params_net_arch": "small",
"params_noise_type": 'normal', 
"value": 0 # only in logs
}

# Alternately, load tuned values  from log dir
#hyper = best_hyperpars(log_dir, env_id, algo)

# Default parameters not tuned
learning_starts=100
optimize_memory_usage=False
# Configured in hyperparams/ddpg.yml
use_sde = True


env = gym.make(env_id)
model = ddpg(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
model = DDPG('MlpPolicy', env, verbose = verbose, tensorboard_log=tensorboard_log, 
            seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed, "vanilla")



