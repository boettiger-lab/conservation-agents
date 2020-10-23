import os
import sys
import gym
from gym import spaces
import gym_fishing
import numpy as np
import torch as th

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise

import leaderboard
url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution


env = gym.make('fishing-v1')
policy_kwargs = dict(activation_fn=th.nn.ReLU, 
                     net_arch=[100, 100, 100, 100, 100], 
                     log_std_init=.22)
model = SAC("MlpPolicy", 
            env, 
            verbose=2, 
            learning_rate=0.001,
            batch_size=16, 
            buffer_size=1000, 
            train_freq=8, 
            tau=0.01, 
            ent_coef=0.004, 
            gamma=0.995, 
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=int(1e6), log_interval=int(1e3))
model.save("tuned-sac")


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/tuned-sac.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard.leaderboard("SAC-tuned", ENV, mean_reward, std_reward, url)



