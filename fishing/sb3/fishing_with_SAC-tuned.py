import gym
import gym_fishing
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import leaderboard
import os
url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution
import pandas as pd

ENV = "fishing-v1"  # Can also do discrete
env = gym.make(ENV)

# SAC defaults:
#learning_rate=0.0003, buffer_size=1000000, learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, n_episodes_rollout=- 1, 
#action_noise=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=- 1, use_sde_at_warmup=False, 
#tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)

hyper = {'gamma': 0.99, 
         'lr': 1.8825727360507924e-05, 
         'batch_size': 512, 
         'buffer_size': 10000, 
         'learning_starts': 10000, 
         'train_freq': 1, 
         'tau': 0.005, 
         'log_std_init': -0.3072998266889968, 
         'net_arch': 'medium'}
policy_kwargs = dict(log_std_init=-3.67, net_arch=[256, 256])

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
            tensorboard_log="/var/log/tensorboard/benchmark")
model.learn(total_timesteps=300000)


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac-tuned.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard.leaderboard("SAC", ENV, mean_reward, std_reward, url)


