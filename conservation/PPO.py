from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, ppo

env_id = "conservation-v5"
algo="ppo"
seed = None
hyper = {'params_batch_size': 32, 
         'params_n_steps': 32,
         'params_gamma': 0.98,
         'params_lr': 8.225772986391458e-05,
         'params_ent_coef': 4.276944290929924e-07, 
         'params_clip_range': 0.2,
         'params_n_epochs': 10,
         'params_gae_lambda': 0.98, 
         'params_max_grad_norm': 0.6,
         'params_vf_coef': 0.3734414549850671, 
         'params_net_arch': 'medium',
         'params_log_std_init': -3.363947481408501, 
         'params_sde_sample_freq': 32, 
         'params_activation_fn': 'relu',
         'value': 50.773604}
         # Best is trial 5 with value: 50.773604
# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- defaults
# hyper = {
# "params_n_steps": 1024,
# "params_batch_size": 64,
# "params_gamma": 0.9999,
# "params_lr": 0.03,
# "params_ent_coef": 4e-08,
# "params_clip_range": 0.1,
# "params_n_epochs": 20,
# "params_gae_lambda": 0.95,
# "params_max_grad_norm": 0.8,
# "params_vf_coef": 0.94,
# "params_sde_sample_freq": -1,
# "params_activation_fn": "relu",
# "params_log_std_init": -2.6,
# "params_net_arch": "small",
# "params_ortho_init": True,
# "value": 0 # only in logs
# }

# Alternately, load tuned values  from log dir
# 0 for best, 1 for second-best, etc
#hyper = best_hyperpars("logs", env_id, algo, 1)

env = make_vec_env(env_id, n_envs = 4, seed = seed)
model = ppo(env, hyper, 'MlpPolicy', verbose = 0, tensorboard_log = "/var/log/tensorboard/single", seed = seed, use_sde = True, device="cpu")
model.learn(total_timesteps = 300000)
custom_eval(model, env_id, algo, seed = seed, outdir = "results", value = hyper["value"])



## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = PPO('MlpPolicy', env, verbose = 0, tensorboard_log="/var/log/tensorboard/single", 
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")


