
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import gym 
import gym_fishing


# ## Continuous action space 
cts_env = gym.make('fishing-v1')


def simul(environment) :
  time_step = np.array([environment.reset()])
  episode_return = np.array([0.0])
  output = np.zeros(shape = (1000, 4))

  for it in range(1000):
    action = max(time_step[0] - 0.5, np.array([0.0]) )
    time_step = environment.step(action)
    episode_return += action
    output[it] = (it, time_step[0], action, episode_return)
  return output 


out=simul(cts_env)

fig, axs = plt.subplots(3,1)
axs[0].plot(out[:,0], out[:,1])
axs[0].set_ylabel('state')
axs[1].plot(out[:,0], out[:,2])
axs[1].set_ylabel('action')
axs[2].plot(out[:,0], out[:,3])
axs[2].set_ylabel('reward')

fig.tight_layout()
plt.savefig("results/optimal_v1.png")



# ## Discrete Action Space

env = gym.make('fishing-v0')
def simulate(environment) :
    time_step = environment.reset()
    episode_return = 0.0
    output = np.zeros(shape = (1000, 4))
    action = 0
    time_step = environment.step(action)
    episode_return += time_step[1]
    output[0] = (0, time_step[0], action, episode_return)

    for it in range(1,1000):
        if time_step[1] <= max(time_step[0] - 0.5, 0.0) :
            action = 1
        if time_step[1] > max(time_step[0] - 0.5, 0.0) :
            action = 2

        time_step = environment.step(action)
        episode_return += time_step[1]
        output[it] = (it, time_step[0], action, episode_return)
    return output    

time_step = env.reset()
time_step = env.step(1)
time_step

out = simulate(env)



fig, axs = plt.subplots(3,1)
axs[0].plot(out[:,0], out[:,1])
axs[0].set_ylabel('state')
axs[1].plot(out[:,0], out[:,2])
axs[1].set_ylabel('action')
axs[2].plot(out[:,0], out[:,3])
axs[2].set_ylabel('reward')

fig.tight_layout()
plt.savefig("results/optimal_v0.png")


