import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import gym_fishing
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

# We use fishing-v1 to test TD3 because it use a continuous action space
env = gym.make('fishing-v1')
model = TD3('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=200000)
model.save("results/TD3")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean reward:", mean_reward, "std:", std_reward)

## Visualize

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
np.savetxt("results/td3.csv", out, delimiter=",")

# ### Vizualisation

results = pd.read_csv('results/td3.csv',
                      names=['time','state','action','reward'])

fig, axs = plt.subplots(3,1)
axs[0].plot(results.time, results.state)
axs[0].set_ylabel('state')
axs[1].plot(results.time, results.action)
axs[1].set_ylabel('action')
axs[2].plot(results.time, results.reward)
axs[2].set_ylabel('reward')

fig.tight_layout()
plt.savefig("results/td3.png")
