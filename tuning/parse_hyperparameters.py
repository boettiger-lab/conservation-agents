import pandas as pd
import os
import shutil
import glob
import numpy as np
from torch import nn as nn
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import gym
import gym_conservation
import gym_fishing

def dir_create(dest):
  """
  Create a directory recursively, removing any existing directory
  """
  if os.path.exists(dest):
    shutil.rmtree(dest)
  os.makedirs(dest)


def best_hyperpars(logs_dir, env_id, algo, i=0):
  """
  Extract the best hyperparameter row from the logs
  """
  reports = glob.glob(os.path.join(logs_dir, algo, "report_" + env_id + "*.csv"))
  df = pd.DataFrame()
  for r in reports:
    df = df.append(pd.read_csv(r))
  best = df.sort_values("value", ascending=False).iloc[i]
  return best


def action_noise(hyper, algo, n_actions):
  """
  Configure Action Noise from hyperparameter logs
  """
  if hyper['params_episodic']:
      hyper['params_n_episodes_rollout'] = 1
      hyper['params_train_freq'], hyper['params_gradient_steps'] = -1, -1
  else:
      hyper['params_train_freq'] = hyper['params_train_freq']
      hyper['params_gradient_steps'] = hyper['params_train_freq']
      hyper['params_n_episodes_rollout'] = -1
  hyper["params_action_noise"] = NormalActionNoise(
      mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions))
  if hyper["params_noise_type"] == "ornstein-uhlenbeck":
    hyper["params_action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['params_noise_std'] * np.ones(n_actions))
  return hyper


def make_policy_kwargs(hyper, algo):
  if hyper["params_net_arch"] == "medium":
    net_arch = [256, 256]
  elif hyper["params_net_arch"] == "big":
    net_arch = [400, 300]
  else:
    net_arch = [64, 64]
  
  ##  NOTE that zoo's tuner always uses separate nets for PPO/A2C actor and critic
  ##  Sometimes that's not actually better, e.g. in fishing! 
  if algo in ["a2c", "ppo"]:
    net_arch = [dict(pi = net_arch, vf = net_arch)]
    
  if "params_activation_fn" in hyper.keys():
    activation_fn = {"tanh": nn.Tanh, 
                     "relu": nn.ReLU, 
                     "elu": nn.ELU, 
                     "leaky_relu": nn.LeakyReLU}[hyper["params_activation_fn"]]
  elif algo in ["a2c", "ppo", "sac"]:
    activation_fn = nn.Tanh
  else:
    activation_fn = nn.ReLU
  if "params_log_std_init" in hyper.keys():
    log_std_init = hyper["params_log_std_init"]
  elif algo in ["a2c", "ppo"]:
    log_std_init = 0.0
  elif algo == "sac":
    log_std_init = -3
  else:
    log_std_init = None
  if algo == "a2c":
    ortho_init = hyper["params_ortho_init"],
  else:
    ortho_init = False
    
  if algo in ["a2c", "ppo"]:
    policy_kwargs = dict(log_std_init = log_std_init,
                         ortho_init = ortho_init,
                         activation_fn = activation_fn,
                         net_arch = net_arch)
  ## technically could pass activation_fn, but tuner does not tune it
  elif algo == "sac":
    policy_kwargs = dict(log_std_init = log_std_init,
                        net_arch = net_arch)
  else:
    policy_kwargs = dict(net_arch = net_arch)
    
    
  return policy_kwargs


# Some parameters, like use_sde, can be configured in initial hyperparameters but are not tuned
# Really we should be reading these from preset hyperparameters yaml
def a2c(env, hyper, policy = "MlpPolicy", tensorboard_log = None, verbose = 1,
        seed = 0, use_sde = True, sde_sample_freq = -1, rms_prop_eps = 1e-05,
        device = "auto"):
  
  policy_kwargs = make_policy_kwargs(hyper, "a2c")
  model = A2C(policy, 
              env, 
              tensorboard_log=tensorboard_log, 
              verbose = verbose, 
              seed = seed,
              use_sde = use_sde,
              sde_sample_freq = sde_sample_freq,
              rms_prop_eps = rms_prop_eps,
              learning_rate = hyper["params_lr"],
              n_steps = np.int(hyper["params_n_steps"]),
              gamma = hyper["params_gamma"],
              gae_lambda = hyper["params_gae_lambda"],
              ent_coef = hyper["params_ent_coef"],
              vf_coef = hyper["params_vf_coef"],
              max_grad_norm = hyper["params_max_grad_norm"],
              use_rms_prop = hyper["params_use_rms_prop"],
              normalize_advantage = hyper["params_normalize_advantage"],
              policy_kwargs = policy_kwargs,
              device = device
          )
  return model



def ppo(env, hyper, policy = "MlpPolicy", 
        verbose = 0, tensorboard_log = None, seed = 0, 
        use_sde = True, device = "auto"):

  policy_kwargs = make_policy_kwargs(hyper, "ppo")
  model = PPO('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log=tensorboard_log, 
              seed = seed,
              use_sde = use_sde,
              n_steps = np.int(hyper["params_n_steps"]),
              batch_size = np.int(hyper["params_batch_size"]),
              gamma = hyper["params_gamma"],
              learning_rate = hyper["params_lr"],
              ent_coef = hyper["params_ent_coef"],
              clip_range = hyper["params_clip_range"],
              n_epochs = np.int(hyper["params_n_epochs"]),
              gae_lambda = hyper["params_gae_lambda"],
              max_grad_norm = hyper["params_max_grad_norm"],
              vf_coef = hyper["params_vf_coef"],
              sde_sample_freq = hyper["params_sde_sample_freq"],
              policy_kwargs = policy_kwargs,
              device = device
          )
  return model




def ddpg(env, hyper, policy = "MlpPolicy", 
        verbose = 0, tensorboard_log = None, seed = 0, 
        use_sde = True, device = "auto"):
  
  policy_kwargs = make_policy_kwargs(hyper, "ddpg")
  hyper = action_noise(hyper, "ddpg", n_actions = env.action_space.shape[0])
  
  model = DDPG('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log = tensorboard_log, 
              seed = seed,
              gamma = hyper['params_gamma'],
              learning_rate = hyper['params_lr'],
              batch_size = np.int(hyper['params_batch_size']),            
              buffer_size = np.int(hyper['params_buffer_size']),
              action_noise = hyper['params_action_noise'],
              train_freq = np.int(hyper['params_train_freq']),
              gradient_steps = np.int(hyper['params_train_freq']),
              n_episodes_rollout = np.int(hyper['params_n_episodes_rollout']),
              policy_kwargs=policy_kwargs,
              device = device)
  return model


def sac(env, hyper, policy = "MlpPolicy", 
        verbose = 0, tensorboard_log = None, seed = 0, 
        use_sde = True, device = "auto"):
                
  policy_kwargs = make_policy_kwargs(hyper, "sac")
  model = SAC('MlpPolicy', 
              env,
              verbose = verbose, 
              tensorboard_log = tensorboard_log,
              seed = seed,
              use_sde = use_sde,
              learning_rate = hyper['params_lr'],
              gamma = hyper['params_gamma'],
              batch_size = np.int(hyper['params_batch_size']),            
              buffer_size = np.int(hyper['params_buffer_size']),
              learning_starts = np.int(hyper['params_learning_starts']),
              train_freq = np.int(hyper['params_train_freq']),
              tau = hyper['params_tau'],
              gradient_steps = np.int(hyper['params_train_freq']), # tuner assumes this
              policy_kwargs=policy_kwargs, 
              device = device)
  return model


def td3(env, hyper, policy = "MlpPolicy", 
        verbose = 0, tensorboard_log = None, seed = 0, 
        use_sde = True, device = "auto"):
 
  policy_kwargs = make_policy_kwargs(hyper, "td3")
  hyper = action_noise(hyper, "td3", n_actions = env.action_space.shape[0])
  
  model = TD3('MlpPolicy', env, 
              verbose = verbose, 
              tensorboard_log = tensorboard_log, 
              seed = seed,
              gamma = hyper['params_gamma'],
              learning_rate = hyper['params_lr'],
              batch_size = np.int(hyper['params_batch_size']),            
              buffer_size = np.int(hyper['params_buffer_size']),
              action_noise = hyper['params_action_noise'],
              train_freq = np.int(hyper['params_train_freq']),
              gradient_steps = np.int(hyper['params_train_freq']),
              n_episodes_rollout = np.int(hyper['params_n_episodes_rollout']),
              policy_kwargs=policy_kwargs,
              device = device)
  return model

AGENT = {"ppo": ppo,
         "a2c": a2c,
         "sac": sac,
         "ddpg": ddpg,
         "td3": td3}

def train_from_logs(algo, env_id, log_dir = "logs", total_timesteps = 300000,
          tensorboard_log = None, seed = 0, verbose = 0,
          n_envs = 4, outdir = "results"):
  
  # create env    
  if(algo in ["a2c", "ppo"]):
    env = make_vec_env(env_id, n_envs = n_envs, seed = seed)
  else:
    env = gym.make(env_id)
  # Create and train agent
  agent = AGENT[algo]
  hyper = best_hyperpars(log_dir, env_id, algo)
  model = agent(env, hyper, 'MlpPolicy', verbose = verbose, tensorboard_log = tensorboard_log, seed = seed)
  model.learn(total_timesteps = total_timesteps)
  # evaluate agent
  custom_eval(model, env_id, algo, seed = seed, outdir = outdir, value = hyper["value"])
  
  
# FIXME env.plot env.simulate are non-standard, should be done with render() loop  
def custom_eval(model, env_id, algo, seed = 0, outdir="results", value = np.nan):
  # eval env
  env = Monitor(gym.make(env_id))
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
  
  ## Simulate
  np.random.seed(seed)
  df = env.simulate(model, reps=10)
  policy = env.policyfn(model, reps=10)
  
  ## Save and print
  print("algo:", algo, "env:", env_id, "mean reward:", mean_reward,
        "std:", std_reward, "tuned_value:", value)
        
  dest = os.path.join(outdir, env_id, algo)
  dir_create(dest)
  model.save(os.path.join(dest, "agent"))
  df.to_csv(os.path.join(dest, "sim.csv"))
  env.plot(df, os.path.join(dest, "sim.png"))
  env.plot_policy(policy, os.path.join(dest, "policy.png"))
  

