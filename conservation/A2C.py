from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, a2c

env_id = "conservation-v6"
algo="a2c"
outdir = "results"
total_timesteps = 3000000
verbose = 0
seed = 0
tensorboard_log="/var/log/tensorboard/single"

# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- A2C defaults
hyper = {
'params_activation_fn':                              "relu",
"params_ent_coef":                               0.00387734,
'params_gae_lambda':                                   0.99,
'params_gamma':                                       0.995,
"params_log_std_init":                              -3.0706,
"params_lr":                                      0.0123908,
"params_lr_schedule":                              "linear",
"params_max_grad_norm":                                   2,
"params_n_steps":                                       128,
"params_net_arch":                                 "medium",
"params_normalize_advantage":                         False,
"params_ortho_init":                                   True,
"params_use_rms_prop":                                 True,
"params_vf_coef":                                  0.972425,
'value':  51.2587}



# Alternately, load tuned values  from log dir
#hyper = best_hyperpars(log_dir, env_id, algo)

# Configured in hyperparams/a2c.yml
use_sde = True
n_envs = 4

env = make_vec_env(env_id, n_envs = n_envs, seed = seed)
model = a2c(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde, device="cpu")
model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])



## Compare to vanilla default execution. 
#model = A2C('MlpPolicy', env, verbose = 0, tensorboard_log=tensorboard_log, 
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")



