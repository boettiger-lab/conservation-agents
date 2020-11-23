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
## SAC #######################################################################

# FIXME read from YAML
hyper = {'gamma': 0.99, 
         'lr': 1.8825727360507924e-05, 
         'batch_size': 512, 
         'buffer_size': 10000, 
         'learning_starts': 10000, 
         'train_freq': 1, 
         'tau': 0.005, 
         'log_std_init': -0.3072998266889968, 
         'net_arch': 'medium'}
policy_kwargs = dict(log_std_init=hyper["log_std_init"], net_arch=[256, 256])

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
            tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)

ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.2)
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
env = gym.make(ENV, C = 0.1, sigma = 0.1)
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









## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c-multi.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/a2c-policy.png")

model.save("results/a2c-multi")
