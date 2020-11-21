import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os
from torch import nn as nn

#file = os.path.basename(__file__)
file = "compute_leaderboard.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/leaderboard"


ENV = "fishing-v1"    
env = gym.make(ENV)


## Constant Escapement ######################################################
model = escapement(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/escapement.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
leaderboard("ESC", ENV, mean_reward, std_reward, url)
print("algo:", "ESC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## MSY ######################################################################
model = msy(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/msy.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
leaderboard("MSY", ENV, mean_reward, std_reward, url)
print("algo:", "MSY", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


# Consider running these in parallel?


## PPO ######################################################################

# load best tuned parameters...

model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("PPO", ENV, mean_reward, std_reward, url)
print("algo:", "PPO", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/ppo-policy.png")


## A2C ######################################################################

# 
# Trial 328 finished with value: 8.025644302368164 and parameters: 
#hyper = {'gamma': 0.98, 'normalize_advantage': False, 'max_grad_norm': 0.3, 'use_rms_prop': True, 'gae_lambda': 0.98, 'n_steps': 16,
#         'lr_schedule': 'linear', 'lr': 0.03490204662520112, 
#         'ent_coef': 0.00026525398345043097, 'vf_coef': 0.18060066335808234, 
#         'log_std_init': -1.1353269076856574, 'ortho_init': True, 'net_arch': 'medium', 'activation_fn': 'relu'}
#policy_kwargs = dict(log_std_init=hyper["log_std_init"],
#                     ortho_init = hyper["ortho_init"],
#                     activation_fn = nn.ReLU,
#                     net_arch=[256, 256])

model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log,
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
leaderboard("A2C", ENV, mean_reward, std_reward, url)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/a2c-policy.png")


## DDPG ######################################################################

# FIXME load best tuned parameters first...

model = DDPG('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
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
            env, verbose=0, 
            use_sde=True,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            learning_starts = hyper['learning_starts'],
            train_freq = hyper['train_freq'],
            tau = hyper['tau'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
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
hyper = {'gamma': 0.95, 'lr': 0.001737794384065678, 'batch_size': 256, 'buffer_size': 1000000, 'episodic': True, 'noise_type': None, 'noise_std': 0.260511264163344, 'net_arch': 'medium'}
policy_kwargs = dict(net_arch=[256, 256])

## WTF are noise_type, noise_std, (action noise? policy noise?) and episodic?  
model = TD3('MlpPolicy', 
            env, 
            verbose=0, 
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
#            noise_type = hyper['noise_type'],
#            noise_std = hyper['noise_std'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("TD3", ENV, mean_reward, std_reward, url)
print("algo:", "TD3", "env:",ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/td3-policy.png")

