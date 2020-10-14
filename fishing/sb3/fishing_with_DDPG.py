import gym
import gym_fishing
import numpy as np
from leaderboard import leaderboard
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import os

ENV = "fishing-v1"
env = gym.make(ENV)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100000, log_interval=100)


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ddpg.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("DDPG", ENV, mean_reward, std_reward, os.path.basename(__file__))

model.save("results/ddpg")
