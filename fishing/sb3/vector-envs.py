import gym
import gym_fishing
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        if rank == 1:
            env = gym.make("fishing-v2", C = 0.2)
        elif rank == 2:
            env = gym.make("fishing-v2", C = 0.1)
        elif rank == 3:
            env = gym.make("fishing-v1")
        elif rank == 0:
            env = gym.make("fishing-v1", r = 0.1)
        env.seed(seed)
        return env
    set_random_seed(seed)
    return _init



if __name__ == '__main__':
    env_id = "fishing-v1"
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    
    model = A2C('MlpPolicy', env, verbose=0, tensorboard_log="/var/log/tensorboard/vec")
    model.learn(total_timesteps=500000)
    model.save("models/a2c-vec")

    
    
    ENV = "fishing-v2"    
    env = gym.make(ENV, C = 0.2)
    model.set_env(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("algo:", "A2C2", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
    ## simulate and plot results for reference
    df = env.simulate(model, reps=10)
    env.plot(df, "results/vec-eval1.png")
    
    ENV = "fishing-v2"    
    env = gym.make(ENV, C = 0.1)
    model.set_env(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
    df = env.simulate(model, reps=10)
    env.plot(df, "results/vec-eval2.png")
    
    
    ENV = "fishing-v1"    
    env = gym.make(ENV)
    model.set_env(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
    df = env.simulate(model, reps=10)
    env.plot(df, "results/vec-eval3.png")
    
    ENV = "fishing-v1"    
    env = gym.make(ENV, r = 0.1)
    model.set_env(env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)
    df = env.simulate(model, reps=10)
    env.plot(df, "results/vec-eval4.png")
    
    



## Simulate from vectorized env: four environments?
# row = []
# obs = env.reset()
# for t in range(base.Tmax):
#   action, _state = model.predict(obs)
#   obs, reward, done, info = env.step(action)
#   fish_population = base.get_fish_population(obs)
#   quota = base.get_quota(action)
#   row.append([t, fish_population, quota, reward])
#   df = pd.DataFrame(row, columns=['time', 'state', 'action', 'reward'])
# 
# 
## Why only one action and one obs, but four different rewards?

