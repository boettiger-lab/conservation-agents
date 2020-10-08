# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''base'': conda)'
#     name: python_defaultSpec_1594304650593
# ---

# # Fishing randomly

# +
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import gym
import gym_fishing

import tensorflow as tf  
from tf_agents.policies import random_tf_policy 
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
# -

# ## Discrete action space

discount = 0.99

# +
env_name = 'fishing-v0'

train_py_env = suite_gym.load(env_name, discount = discount)
eval_py_env = suite_gym.load(env_name, discount = discount)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
# -

# I create a policy that will randomly select an action for each timestep.

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


def simulate(environment, policy):
  total_return = 0.0
  time_step = environment.reset()
  episode_return = 0.0

  output = np.zeros(shape = (1000, 4))
  for it in range(1000):
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
    episode_return += time_step.reward
    output[it] = (it, time_step.observation, action_step.action, episode_return)

  return output


out = simulate(eval_env, random_policy)

plt.plot(out[:,1])
plt.ylabel('state')
plt.show()

plt.plot(out[:,2])
plt.ylabel('action')
plt.show()

plt.plot(out[:,3])
plt.ylabel('reward')
plt.show()

# ## Continuous action space

# +
env_name = 'fishing-v1'

train_py_env = suite_gym.load(env_name, discount = discount)
eval_py_env = suite_gym.load(env_name, discount = discount)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
# -

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

out = simulate(eval_env, random_policy)



fig, axs = plt.subplots(3,1)
axs[0].plot(out[:,0], out[:,1])
axs[0].set_ylabel('state')
axs[1].plot((out[:,0], out[:,2])
axs[1].set_ylabel('action')
axs[2].plot((out[:,0], out[:,3])
axs[2].set_ylabel('reward')

fig.tight_layout()
plt.savefig("results/random.png")



