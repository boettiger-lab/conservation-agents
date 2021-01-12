from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, td3

import gym, gym_conservation
env_id = "conservation-v6"  #"fishing-v1"
algo = "td3"
outdir = "results"
total_timesteps = 300000
verbose = 0
seed = 0
tensorboard_log="/var/log/tensorboard/single"
log_dir = "logs"

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- defaults
hyper = {
"params_gamma": 0.99,
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
hyper = best_hyperpars(log_dir, env_id, algo)
print(hyper)

# Default parameters not tuned
learning_starts=100
optimize_memory_usage=False
# Configured in hyperparams/td3.yml
use_sde = True
tau = 0.005
action_noise=None
target_noise_clip=0.5
policy_delay=2
target_policy_noise=0.2

env = gym.make(env_id)
model = td3(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = TD3('MlpPolicy', env, verbose = 0, tensorboard_log = tensorboard_log, 
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")



