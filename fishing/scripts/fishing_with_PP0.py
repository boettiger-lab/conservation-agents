# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # GYM FISHING WITH PP0

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import gym
import gym_fishing

from stable_baselines3 import PPO

# + tags=[]
env = gym.make('fishing-v0')
env.n_actions = 100
model = PPO('MlpPolicy', env, verbose=1)

# + tags=["outputPrepend"]
model.learn(total_timesteps=200000)


# -

def simulate(environment, model):
  obs = env.reset()
  episode_return = 0.0
  output = np.zeros(shape = (1000, 4))

  for it in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)  
    episode_return += reward
    output[it] = (it, obs, action, episode_return)
  return output


out = simulate(env, model)
np.savetxt("ppo.csv", out, delimiter=",")

# ### Vizualisation

datapath = '/Users/kevinab/Desktop/PRe/conservation-agents/ppo.csv'
results = pd.read_csv(datapath, names=['time','state','harvest','action'])

plt.plot(results.iloc[:,1])
plt.ylabel('state')
plt.show()

plt.plot(results.iloc[:,2])
plt.ylabel('action')
plt.show()

plt.plot(results.iloc[:,3])
plt.ylabel('reward')
plt.show()


