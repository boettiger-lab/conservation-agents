import gym
import gym_fishing
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, 
                       r = (rank + 1) / 10)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "fishing-v1"
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    
    model = A2C('MlpPolicy', env, verbose=0, tensorboard_log="/var/log/tensorboard/vec", device = "cpu")
    model.learn(total_timesteps=300000)
    model.save("models/a2c-vec")

    base = gym.make(env_id, r = 0.1)
    
    
    ## simulate and plot results
    df = base.simulate(model, reps=10)
    base.plot(df, "results/vec-a2c.png")







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

