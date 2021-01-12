from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, ppo

env_id = "conservation-v6"
algo="ppo"
seed = None
tensorboard_log="/var/log/tensorboard/single"
 
hyper = {
'params_activation_fn':                            'relu',
'params_batch_size':                                  128,
'params_clip_range':                                  0.4,
'params_ent_coef':                            1.52337e-07,
'params_gae_lambda':                                 0.99,
'params_gamma':                                     0.999,
'params_log_std_init':                           -2.82008,
'params_lr':                                  0.000929325,
'params_max_grad_norm':                               0.5,
'params_n_epochs':                                     20,
'params_n_steps':                                      32,
'params_net_arch':                                'medium',
'params_sde_sample_freq':                              64,
'params_vf_coef':                                0.261185,
'value': 51.3641}
 
# Alternately, load tuned values  from log dir
# 0 for best, 1 for second-best, etc

#hyper = best_hyperpars("logs", env_id, algo, 0)

env = make_vec_env(env_id, n_envs = 4, seed = seed)
model = ppo(env, hyper, 'MlpPolicy', verbose = 0, tensorboard_log = tensorboard_log, seed = seed, use_sde = True, device="cpu")
model.learn(total_timesteps = 300000)
custom_eval(model, env_id, algo, seed = seed, outdir = "results", value = hyper["value"])



## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = PPO('MlpPolicy', env, verbose = 0, tensorboard_log = tensorboard_log, 
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")


