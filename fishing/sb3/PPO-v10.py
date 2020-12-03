import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os


file = os.path.basename(__file__)
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/single"

ENV = "fishing-v10"    
env = gym.make(ENV)
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log )
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print("algo:", "PPO", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/PPO-v4.png")
