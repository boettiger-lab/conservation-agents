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


ENV = "fishing-v1"    
env = gym.make(ENV)
model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.set_env(env)

model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


ENV = "fishing-v1"    
env = gym.make(ENV)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)




ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)



ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.4)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.2)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.1, sigma = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)






## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c-multi.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/a2c-policy.png")

model.save("results/a2c-multi")
