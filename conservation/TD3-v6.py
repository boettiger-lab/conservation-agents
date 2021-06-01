from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

import gym, gym_conservation
env_id = "conservation-v6"  #"fishing-v1"
algo = "td3"
outdir = "results"
total_timesteps = 1500000
verbose = 0
seed = 0
tensorboard_log="/var/log/tensorboard/single"
log_dir = "logs"

noise_std = 0.4805935357322933,
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=noise_std * np.ones(1))
hyper = {
"gamma": 0.995,
"learning_rate":  8.315382409902049e-05,
"batch_size": 512,
"buffer_size": 10000,
"train_freq": 1000,
"gradient_steps": 1000,
"n_episodes_rollout": -1,
"action_noise": action_noise,
"policy_kwargs": {"net_arch": [64,64]}
}

#norm_env = VecNormalize(make_vec_env(env_id), gamma = hyper["gamma"])
env = gym.make(env_id)

model = TD3('MlpPolicy', env, verbose = verbose, tensorboard_log = tensorboard_log, seed = seed, **hyper)
model.learn(total_timesteps = total_timesteps)

# Simulate a run with the trained model, visualize result
df = env.simulate(model, reps=5)
env.plot(df, "td3_sim.png")
model.save("td3-v6")

env.plot_policy(df, "td3_policy.png")

##Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=500)
print("mean reward:", mean_reward, "std:", std_reward)
