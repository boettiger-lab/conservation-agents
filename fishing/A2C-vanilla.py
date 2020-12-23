from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append("../tuning")
from parse_hyperparameters import learn, best_hyperpars, custom_eval

env_id = "fishing-v1"
algo="a2c"
outdir = "vanilla"
total_timesteps = 100000
verbose = 1
seed = 0
tensorboard_log = None

## Configured in hyperparams/a2c.yml
use_sde = True
n_envs = 4

env = make_vec_env(env_id, n_envs=n_envs, seed = seed)
model = A2C('MlpPolicy', 
            env, 
            verbose = verbose, 
            tensorboard_log=tensorboard_log, 
            seed = seed,
            use_sde = use_sde)

model.learn(total_timesteps = total_timesteps)
custom_eval(model, env_id, algo, seed, outdir)
