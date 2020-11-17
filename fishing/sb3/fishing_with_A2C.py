import numpy as np
import gym
import gym_fishing
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import leaderboard
import os
url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v1" # A2C can do discrete & cts
env = gym.make(ENV)
model = A2C('MlpPolicy', env, verbose=0, tensorboard_log="/var/log/tensorboard/benchmark")
model.learn(total_timesteps=300000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")



base = gym.make(ENV, init_state = 0.3)
df = base.simulate(model, reps=10)
base.plot(df, "results/a2c-rebuild.png")


## Evaluate model
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
#leaderboard.leaderboard("A2C", ENV, mean_reward, std_reward, url)

#model.save("models/a2c")
#print("mean reward:", mean_reward, "std:", std_reward)
