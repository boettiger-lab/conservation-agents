import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

file = os.path.basename(__file__)
#file = "multi_env.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/multi-env"

env = gym.make("fishing-v1")
model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)



ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.2)
model.set_env(env)
model.learn(total_timesteps=500000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
print("algo:", "A2C2", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/serial-eval1.png")

ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.1)
model.set_env(env)
model.learn(total_timesteps=500000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
df = env.simulate(model, reps=10)
env.plot(df, "results/serial-eval2.png")


ENV = "fishing-v1"    
env = gym.make(ENV)
model.set_env(env)
model.learn(total_timesteps=500000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
df = env.simulate(model, reps=10)
env.plot(df, "results/serial-eval3.png")

ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
model.set_env(env)
model.learn(total_timesteps=500000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
df = env.simulate(model, reps=10)
env.plot(df, "results/serial-eval4.png")






