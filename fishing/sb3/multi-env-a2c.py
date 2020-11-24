import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import A2C, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

file = os.path.basename(__file__)
#file = "multi_env.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/multi-env"

env = gym.make("fishing-v1")
## A2C #######################################################################

# 
# Trial 328 finished with value: 8.025644302368164 and parameters: 
hyper = {'gamma': 0.98, 'normalize_advantage': False, 'max_grad_norm': 0.3,
         'use_rms_prop': True, 'gae_lambda': 0.98, 'n_steps': 16,
         'lr_schedule': 'linear', 'lr': 0.03490204662520112, 
         'ent_coef': 0.00026525398345043097, 'vf_coef': 0.18060066335808234, 
         'log_std_init': -1.1353269076856574, 'ortho_init': True, 'net_arch': 'medium', 'activation_fn': 'relu'}
policy_kwargs = dict(log_std_init=hyper["log_std_init"],
                     ortho_init = hyper["ortho_init"],
                     activation_fn = nn.ReLU,
                     net_arch=[256, 256])

model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log,
            gamma = hyper["gamma"],
            learning_rate = hyper["lr"],
            normalize_advantage = hyper["normalize_advantage"],
            gae_lambda = hyper["gae_lambda"],
            n_steps = hyper["n_steps"],
            ent_coef = hyper["ent_coef"],
            vf_coef = hyper["vf_coef"],
            policy_kwargs = policy_kwargs
            )


ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.2)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v2-c02.png")

ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.4)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v2-c04.png")


ENV = "fishing-v2"    
env = gym.make(ENV, C = 0.1, sigma = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v2-c01-s01.png")


ENV = "fishing-v1"    
env = gym.make(ENV)
model.set_env(env)

model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v1.png")


ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v1-r01.png")


ENV = "fishing-v1"    
env = gym.make(ENV)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v1b.png")




ENV = "fishing-v1"    
env = gym.make(ENV, r = 0.1)
model.set_env(env)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/A2C-v1-r01b.png")



model.save("results/A2C-env")
