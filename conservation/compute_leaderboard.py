import gym
import gym_conservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import sys
sys.path.append("../tuning")
from parse_hyperparameters import ppo_best, a2c_best, td3_best, sac_best, ddpg_best

tensorboard_log="/var/log/tensorboard/leaderboard"

seed = 0

ENV = "conservation-v5"    
env = gym.make(ENV)
vec_env = make_vec_env(ENV, n_envs=4, seed=seed) # parallel workers for PPO, A2C



## PPO ######################################################################

# load best tuned parameters...
model = ppo_best('MlpPolicy', vec_env, verbose=0, tensorboard_log=tensorboard_log, seed = seed)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print("algo:", "PPO", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/ppo-policy.png")
model.save("models/ppo-leaderboard")


## A2C ######################################################################


model = a2c_best('MlpPolicy', vec_env, verbose=0, tensorboard_log=tensorboard_log, seed = seed)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/a2c-policy.png")
model.save("models/a2c-tuned")




## DDPG ######################################################################


model = ddpg_best('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("algo:", "DDPG", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/ddpg.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/ddpg-policy.png")
model.save("models/ddpg-tuned")

## SAC #######################################################################
model = sac_best('MlpPolicy', 
            env, verbose=0, tensorboard_log=tensorboard_log, seed = seed, use_sde=True)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print("algo:", "SAC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/sac-policy.png")
model.save("models/sac-tuned")


## TD3 ######################################################################

model = td3_best('MlpPolicy', env,  verbose=0, tensorboard_log=tensorboard_log, seed = seed,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            action_noise = hyper['action_noise'],
            train_freq = hyper['train_freq'],
            gradient_steps = hyper['train_freq'],
            n_episodes_rollout = hyper['n_episodes_rollout'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# Rescale score against optimum solution in this environment
opt = escapement(env)
opt_reward, std_reward = evaluate_policy(opt, env, n_eval_episodes=100)
mean_reward = mean_reward / opt_reward; std_reward = std_reward / opt_reward   
leaderboard("TD3", ENV, mean_reward, std_reward, url)
print("algo:", "TD3", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")
policy = env.policyfn(model, reps=10)
env.plot_policy(policy, "results/td3-policy.png")
model.save("models/td3-tuned")
