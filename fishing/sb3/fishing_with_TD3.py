import gym
import gym_fishing
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
import leaderboard
import os
url = leaderboard.hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v1" # TD3 can do continuous action spaces only
env = gym.make(ENV)
model = TD3('MlpPolicy', env, verbose=0, tensorboard_log="/var/log/tensorboard/benchmark")
model.learn(total_timesteps=300000)
#model.load("models/td3")

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard.leaderboard("TD3", ENV, mean_reward, std_reward, url)


model.save("models/td3")


