import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

#file = os.path.basename(__file__)
file = "compute_leaderboard.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/leaderboard"

seed = 0

ENV = "fishing-v2"    
env = gym.make(ENV)
vec_env = make_vec_env(ENV, n_envs=4, seed=seed) # parallel workers for PPO, A2C


## Constant Escapement ######################################################
model = escapement(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/escapement.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
# Rescale score against optimum solution in this environment (err... identical in this case)
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward 
leaderboard("ESC", ENV, mean_reward, std_reward, url)
print("algo:", "ESC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## MSY ######################################################################
model = msy(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/msy.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward 
leaderboard("MSY", ENV, mean_reward, std_reward, url)
print("algo:", "MSY", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


# Consider running these in parallel?


## PPO ######################################################################

# load best tuned parameters...
model = PPO('MlpPolicy', vec_env, verbose=0, tensorboard_log=tensorboard_log, seed = seed)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   

leaderboard("PPO", ENV, mean_reward, std_reward, url)
print("algo:", "PPO", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/ppo-policy.png")


## A2C ######################################################################

# ought to be vectorized!
# Trial 328 finished with value: 8.025644302368164 and parameters: 
hyper = {'gamma': 0.98, 'normalize_advantage': False, 'max_grad_norm': 0.3,
         'use_rms_prop': True, 'gae_lambda': 0.98, 'n_steps': 16,
         'lr_schedule': 'linear', 'lr': 0.03490204662520112, 
         'ent_coef': 0.00026525398345043097, 'vf_coef': 0.18060066335808234, 
         'log_std_init': -1.1353269076856574, 'ortho_init': True, 'net_arch': 'medium', 'activation_fn': 'relu'}
policy_kwargs = dict(log_std_init=hyper["log_std_init"],
                     ortho_init = hyper["ortho_init"],
                     activation_fn = nn.ReLU,
                     net_arch=[256, 256])

model = A2C('MlpPolicy', vec_env, verbose=0, tensorboard_log=tensorboard_log, seed = seed,
#            gamma = hyper["gamma"],
#            learning_rate = hyper["lr"],
#            normalize_advantage = hyper["normalize_advantage"],
#            gae_lambda = hyper["gae_lambda"],
#            n_steps = hyper["n_steps"],
#            ent_coef = hyper["ent_coef"],
#            vf_coef = hyper["vf_coef"],
#            policy_kwargs = policy_kwargs
            )
            
model.learn(total_timesteps=300000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("A2C", ENV, mean_reward, std_reward, url)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/a2c-policy.png")


## DDPG ######################################################################

#Trial 15 finished with value: 7.6994757652282715 and parameters: 
hyper = {'gamma': 0.995, 'lr': 1.03681319842628e-05, 'batch_size': 128, 
         'buffer_size': 1000000, 'episodic': False, 'train_freq': 2000, 
         'noise_type': 'normal', 'noise_std': 0.513787888663763, 
         'net_arch': 'medium'}
policy_kwargs = dict(net_arch=[256, 256]) # medium
if hyper['episodic']:
    hyper['n_episodes_rollout'] = 1
    hyper['train_freq'], hyper['gradient_steps'] = -1, -1
else:
    hyper['train_freq'] = hyper['train_freq']
    hyper['gradient_steps'] = hyper['train_freq']
    hyper['n_episodes_rollout'] = -1

n_actions = env.action_space.shape[0]
if hyper["noise_type"] == "normal":
    hyper["action_noise"] = NormalActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )
elif noise_type == "ornstein-uhlenbeck":
    hyper["action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )

model = DDPG('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            action_noise = hyper['action_noise'],
            train_freq = hyper['train_freq'],
            gradient_steps = hyper['train_freq'],
            n_episodes_rollout = hyper['n_episodes_rollout'],
            policy_kwargs=policy_kwargs)
model = DDPG('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("DDPG", ENV, mean_reward, std_reward, url)
print("algo:", "DDPG", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/ddpg.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/ddpg-policy.png")

## SAC #######################################################################

# FIXME read from YAML
hyper = {'gamma': 0.99, 
         'lr': 1.8825727360507924e-05, 
         'batch_size': 512, 
         'buffer_size': 10000, 
         'learning_starts': 10000, 
         'train_freq': 1, 
         'tau': 0.005, 
         'log_std_init': -0.3072998266889968, 
         'net_arch': 'medium'}
policy_kwargs = dict(log_std_init=hyper["log_std_init"], net_arch=[256, 256])

model = SAC('MlpPolicy', 
            env, verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            use_sde=True,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            learning_starts = hyper['learning_starts'],
            train_freq = hyper['train_freq'],
            tau = hyper['tau'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000)

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("SAC", ENV, mean_reward, std_reward, url)
print("algo:", "SAC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/sac-policy.png")


## TD3 ######################################################################

# load best tuned parameters... # FIXME read from csv?

# [I 2020-11-20 19:27:53,592] Trial 2 finished with value: 8.015005111694336 and parameters: 
hyper = {'gamma': 0.95, 'lr': 0.001737794384065678, 'batch_size': 256, 
         'buffer_size': 1000000, 'episodic': True, 'noise_type': None, 
         'noise_std': 0.260511264163344, 'net_arch': 'medium'}
policy_kwargs = dict(net_arch=[256, 256])

# [I 2020-11-24 18:03:25,390] Trial 29 finished with value: 7.736424446105957 and parameters: 
# {'gamma': 0.99, 'lr': 3.550974876596194e-05, 'batch_size': 64, 'buffer_size': 100000, 'episodic': False, 'train_freq': 128, 'noise_type': 'ornstein-uhlenbeck', 'noise_std': 0.6615816256241758, 'net_arch': 'small'}. Best is trial 29 with value: 7.736424446105957.

#Trial 15 finished with value: 7.6994757652282715 and parameters: 
hyper = {'gamma': 0.995, 'lr': 0.0001355522450968401, 'batch_size': 128, 
         'buffer_size': 10000, 'episodic': False, 'train_freq': 128, 
         'noise_type': 'normal', 'noise_std': 0.6656948079225263, 
         'net_arch': 'big'}
policy_kwargs = dict(net_arch=[400, 300]) # big



if hyper['episodic']:
    hyper['n_episodes_rollout'] = 1
    hyper['train_freq'], hyper['gradient_steps'] = -1, -1
else:
    hyper['train_freq'] = hyper['train_freq']
    hyper['gradient_steps'] = hyper['train_freq']
    hyper['n_episodes_rollout'] = -1


n_actions = env.action_space.shape[0]
if hyper["noise_type"] == "normal":
    hyper["action_noise"] = NormalActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )
elif noise_type == "ornstein-uhlenbeck":
    hyper["action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )

model = TD3('MlpPolicy', env,  verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            action_noise = hyper['action_noise'],
            train_freq = hyper['train_freq'],
            gradient_steps = hyper['train_freq'],
            n_episodes_rollout = hyper['n_episodes_rollout'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("TD3", ENV, mean_reward, std_reward, url)
print("algo:", "TD3", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/td3-policy.png")

