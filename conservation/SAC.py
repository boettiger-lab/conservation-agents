from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import sys
sys.path.append("../tuning")
from parse_hyperparameters import best_hyperpars, custom_eval, sac

import gym
import gym_conservation

env_id = "conservation-v6"
algo="sac"
outdir = "results"
total_timesteps = 1500000
verbose = 0
seed = 0
tensorboard_log="/var/log/tensorboard/single"
log_dir = "logs"
# NB: See utils/hyperparams_opt.py for what is and isn't tuned for each model!
# Manual -- A2C defaults
hyper = {
"params_batch_size":                                  256,
"params_buffer_size":                               10000,
"params_gamma":                                    0.9999,
"params_learning_starts":                           10000,
"params_log_std_init":                           0.600596,
"params_lr":                                  0.000958776,
"params_net_arch":                                    "big",
"params_tau":                                       0.005,
"params_train_freq":                                   64,
"value": 161.096 # only in logs
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
hyper = best_hyperpars(log_dir, env_id, algo)
print(hyper)
# Override defaults
use_sde = True


env = make_vec_env(env_id, n_envs=1) # Zoo still wraps, even when n_env=1
model = sac(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, use_sde = use_sde)
model.learn(total_timesteps = total_timesteps)
# eval does not use normalized env, so it reports true rewards
custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])


## Compare to vanilla default execution.  Vanilla is no action noise, but tuning always uses action noise(?)
#model = SAC('MlpPolicy', env, verbose = 0, tensorboard_log = tensorboard_log,
#            seed = seed, use_sde = use_sde)
#model.learn(total_timesteps = total_timesteps)
#custom_eval(model, env_id, algo, seed, "vanilla")
