import gym
import gym_fishing
import numpy as np
import leaderboard
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import os

url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v1" # DDPG can do cts action spaces only
env = gym.make(ENV)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=0, tensorboard_log="/var/log/tensorboard/benchmark")
model.learn(total_timesteps=300000, log_interval=100)


## "mini-transfer learning": test with different initial condition
env = gym.make(ENV)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ddpg.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
leaderboard.leaderboard("DDPG", ENV, mean_reward, std_reward, url)

model.save("results/ddpg")
print("mean reward:", mean_reward, "std:", std_reward)
