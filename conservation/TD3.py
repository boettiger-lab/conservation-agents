import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import gym_conservation

seed = 0

env = gym.make("conservation-v2", init_state = 0.1, sigma = 0.05)

## TD3 ######################################################################
hyper = {'gamma': 0.999, 'lr': 0.0001, 'batch_size': 128, 
         'buffer_size': 10000, 'episodic': False, 'train_freq': 12, 
         'noise_type': 'normal', 'noise_std': 10, 
         'net_arch': 'big'}
policy_kwargs = dict(net_arch=[256, 256]) # medium



if hyper['episodic']:
    hyper['n_episodes_rollout'] = 1
    hyper['train_freq'], hyper['gradient_steps'] = -1, -1
else:
    hyper['train_freq'] = hyper['train_freq']
    hyper['gradient_steps'] = hyper['train_freq']
    hyper['n_episodes_rollout'] = -1


n_actions = env.action_space.shape[0]
if hyper["noise_type"] == "normal":
    hyper["action_noise"] = NormalActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )
elif hyper["noise_type"] == "ornstein-uhlenbeck":
    hyper["action_noise"] = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma= hyper['noise_std'] * np.ones(n_actions)
    )

model = TD3('MlpPolicy', env,  verbose=0, seed = seed,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            action_noise = hyper['action_noise'],
            train_freq = hyper['train_freq'],
            gradient_steps = hyper['train_freq'],
            n_episodes_rollout = hyper['n_episodes_rollout'],
            policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1e6)



## Simulate a run with the trained model, visualize result
df = env.simulate(model)
env.plot(df, "results/TD3.png")

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
