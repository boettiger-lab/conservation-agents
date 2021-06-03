from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, ppo

env_id = "fishing-v1"
algo="ppo"
outdir = "results"
total_timesteps = 100000
verbose = 0
seed = 0
tensorboard_log = None

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- defaults
hyper = {
"params_n_steps": 2048,
"params_batch_size": 64,
"params_gamma": 0.99,
"params_lr": 0.0003,
"params_ent_coef": 0.0,
"params_clip_range": 0.2,
"params_n_epochs": 10,
"params_gae_lambda": 0.95,
"params_max_grad_norm": 0.5,
"params_vf_coef": 0.5,
"params_sde_sample_freq": -1,
"params_activation_fn": "tanh",
"params_log_std_init": 0.0,
"params_net_arch": "small",
"params_ortho_init": True,
"value": 0 # only in logs
}

# Alternately, load tuned values  from log dir
#hyper = best_hyperpars(log_dir, env_id, algo)

# Default parameters not tuned
target_kl = None

# Configured in hyperparams/ppo.yml
use_sde = True
n_envs = 4


env = make_vec_env(env_id, n_envs = n_envs, seed = seed)
model = ppo(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution
model = PPO('MlpPolicy', env, verbose = verbose, tensorboard_log=tensorboard_log, 
            seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed, "vanilla")



