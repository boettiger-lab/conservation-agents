import pandas as pd
import os
import glob
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN

logs_dir = "logs"
algos = os.listdir(logs_dir)


def best_hyperpars(logs_dir, algo):
  """
  Extract the best hyperparameter row from the logs
  """
  reports = glob.glob(os.path.join(logs_dir, algo, "report*.csv"))
  df = pd.DataFrame()
  for r in reports:
    df = df.append(pd.read_csv(r))
    best = df.iloc[df['value'].idxmax()]
  return best  
    

# Some parameters, like use_sde, can be configured in initial hyperparameters but are not tuned
# Really we should be reading these from preset hyperparameters yaml
def a2c_best(policy, env, logs_dir = "logs", 
             verbose = 0, tensorboard_log = None, seed = None, use_sde = True):
  
  hyper = best_hyperpars(logs_dir, "a2c")
  
  
  activation_fn = {"tanh": nn.Tanh, 
                   "relu": nn.ReLU, 
                   "elu": nn.ELU, 
                   "leaky_relu": nn.LeakyReLU}[hyper["params_activation_fn"]]

  
  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]
  
  policy_kwargs = dict(log_std_init=hyper["params_log_std_init"],
                     ortho_init = hyper["params_ortho_init"],
                     activation_fn = activation_fn,
                     net_arch = net_arch)
  
  model = A2C('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log=tensorboard_log, 
              seed = seed,
              learning_rate = hyper["params_lr"],
              n_steps = hyper["params_n_steps"],
              gamma = hyper["params_gamma"],
              gae_lambda = hyper["params_gae_lambda"],
              ent_coef = hyper["params_ent_coef"],
              vf_coef = hyper["params_vf_coef"],
              max_grad_norm = hyper["params_max_grad_norm"],
              use_rms_prop = hyper["params_use_rms_prop"],
              use_sde = use_sde,
              normalize_advantage = hyper["params_normalize_advantage"],
              policy_kwargs = policy_kwargs
          )
  return model



def ppo_best(policy, env, logs_dir = "logs", 
             verbose = 0, tensorboard_log = None, seed = None, use_sde = True):
  
  hyper = best_hyperpars(logs_dir, "ppo")
  
  activation_fn = {"tanh": nn.Tanh, 
                   "relu": nn.ReLU, 
                   "elu": nn.ELU, 
                   "leaky_relu": nn.LeakyReLU}[hyper["params_activation_fn"]]

  
  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]
  
  policy_kwargs = dict(log_std_init=hyper["params_log_std_init"],
                     activation_fn = activation_fn,
                     net_arch = net_arch)
  
  model = PPO('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log=tensorboard_log, 
              seed = seed,
              learning_rate = hyper["params_lr"],
              n_epochs = hyper["params_n_epochs"],
              n_steps = hyper["params_n_steps"],
              gamma = hyper["params_gamma"],
              gae_lambda = hyper["params_gae_lambda"],
              ent_coef = hyper["params_ent_coef"],
              vf_coef = hyper["params_vf_coef"],
              max_grad_norm = hyper["params_max_grad_norm"],
              batch_size = hyper["params_batch_size"],
              clip_range = hyper["params_clip_range"],
              use_sde = use_sde,
              sde_sample_freq = hyper["params_sde_sample_freq"],
              policy_kwargs = policy_kwargs
          )
  return model



def ddpg_best(policy, env, logs_dir = "logs", 
              verbose=0, tensorboard_log=None, seed=None):
  
  hyper = best_hyperpars(logs_dir, "ddpg")
  
#  if hyper["params_activation_fn"] == "relu":
#    activation_fn = nn.ReLU
  
  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]

  policy_kwargs = dict(net_arch= net_arch)
  
  if hyper['params_episodic']:
      hyper['params_n_episodes_rollout'] = 1
      hyper['params_train_freq'], hyper['params_gradient_steps'] = -1, -1
  else:
      hyper['params_train_freq'] = hyper['params_train_freq']
      hyper['params_gradient_steps'] = hyper['params_train_freq']
      hyper['params_n_episodes_rollout'] = -1
  
  n_actions = env.action_space.shape[0]
  hyper["params_action_noise"] = NormalActionNoise(
      mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions)
  )
  if noise_type == "ornstein-uhlenbeck":
    hyper["params_action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions)
    )

  model = DDPG('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log = tensorboard_log, 
              seed = seed,
              gamma = hyper['params_gamma'],
              learning_rate = hyper['params_lr'],
              batch_size = hyper['params_batch_size'],            
              buffer_size = hyper['params_buffer_size'],
              action_noise = hyper['params_action_noise'],
              train_freq = hyper['params_train_freq'],
              gradient_steps = hyper['params_train_freq'],
              n_episodes_rollout = hyper['params_n_episodes_rollout'],
              policy_kwargs=policy_kwargs)
  return model




def sac_best(policy, env, logs_dir = "logs", 
              verbose = 0, tensorboard_log = None, seed = None,
              use_sde = True):
  
  hyper = best_hyperpars(logs_dir, "sac")

  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]

      
  policy_kwargs = dict(log_std_init = hyper["params_log_std_init"], 
                       net_arch = net_arch)
  model = SAC('MlpPolicy', 
              env,
              verbose = verbose, 
              tensorboard_log = tensorboard_log,
              seed = seed,
              use_sde = use_sde,
              learning_rate = hyper['params_lr'],
              gamma = hyper['params_gamma'],
              batch_size = hyper['params_batch_size'],            
              buffer_size = hyper['params_buffer_size'],
              learning_starts = hyper['params_learning_starts'],
              train_freq = hyper['params_train_freq'],
              tau = hyper['params_tau'],
              policy_kwargs=policy_kwargs)
  return model
  
              





def td3_best(policy, env, logs_dir = "logs", 
              verbose = 0, tensorboard_log = None, seed = None):
  
  hyper = best_hyperpars(logs_dir, "td3")
  
#  if hyper["params_activation_fn"] == "relu":
#    activation_fn = nn.ReLU
  
  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]

  policy_kwargs = dict(net_arch= net_arch)
  
  if hyper['params_episodic']:
      hyper['params_n_episodes_rollout'] = 1
      hyper['params_train_freq'], hyper['params_gradient_steps'] = -1, -1
  else:
      hyper['params_train_freq'] = hyper['params_train_freq']
      hyper['params_gradient_steps'] = hyper['params_train_freq']
      hyper['params_n_episodes_rollout'] = -1
  
  ## Normal action noise is default, OU only if requested
  n_actions = env.action_space.shape[0]
  hyper["params_action_noise"] = NormalActionNoise(
      mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions)
  )
  if noise_type == "ornstein-uhlenbeck":
    hyper["params_action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions)
    )

  model = TD3('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log = tensorboard_log, 
              seed = seed,
              gamma = hyper['params_gamma'],
              learning_rate = hyper['params_lr'],
              batch_size = hyper['params_batch_size'],            
              buffer_size = hyper['params_buffer_size'],
              action_noise = hyper['params_action_noise'],
              train_freq = hyper['params_train_freq'],
              gradient_steps = hyper['params_train_freq'],
              n_episodes_rollout = hyper['params_n_episodes_rollout'],
              policy_kwargs=policy_kwargs)
  return model
  
  
  

