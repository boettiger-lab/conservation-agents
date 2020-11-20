import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os
url = hash_url(os.path.basename(__file__)) # get hash URL at start of execution

ENV = "fishing-v1" # can do cts or discrete
env = gym.make(ENV)
model = escapement(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/escapement.png")
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/escapement-policy.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
leaderboard.leaderboard("ESC", ENV, mean_reward, std_reward, url)

########################################################
model = msy(env)
## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/msy.png")
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/msy-policy.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
leaderboard.leaderboard("MSY", ENV, mean_reward, std_reward, url)

