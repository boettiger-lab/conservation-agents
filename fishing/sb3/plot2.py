import gym
import gym_fishing
from stable_baselines3 import SAC
import numpy as np

env = gym.make('fishing-v1', r = 0.1, init_state=1.0)
for i in range(5):
    model = SAC.load(f"models/sb3_sac_"+str(i))

    ## simulate and plot results
    df = env.simulate(model, reps=25)
    env.plot(df, "results/sb3_sac_" + str(i) + ".png")


      
