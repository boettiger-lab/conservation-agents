import gym
import gym_fishing
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard
import os
url = hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v0"
env = gym.make(ENV, n_actions = 100)
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=400000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("PPO", ENV, mean_reward, std_reward, url)


model.save("models/ppo")
