import gym
import gym_fishing
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('fishing-v0', n_actions = 100)
               
model = A2C('MlpPolicy', env, verbose=2)
model.learn(total_timesteps=200000)
model.save("results/a2c")


# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean reward:", mean_reward, "std:", std_reward)

# delete & re-load trained model to demonstrate loading
del model 


model = A2C.load("results/a2c")
env = gym.make('fishing-v0', 
               file = "results/a2c.csv", 
               fig = "results/a2c.png",
               n_actions = 100)

## Visualize a single simulation
obs = env.reset()
for i in range(100):
  action, _state = model.predict(obs)
  obs, reward, done, info = env.step(action)
  env.render()
env.plot()


