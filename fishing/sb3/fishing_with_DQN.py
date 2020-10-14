## Fishing with DQN example
import gym
import gym_fishing
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard
import os

# Create environment
ENV = "fishing-v0"
env = gym.make(ENV, n_actions=100)
# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=0)
# Train the agent
model.learn(total_timesteps=int(1e5))

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/dqn.png")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)
leaderboard("DQN", ENV, mean_reward, std_reward, os.path.basename(__file__))

# Save the agent
model.save("models/dqn_fish_v0")
