from stable_baselines import PPO2
import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import make_vec_env


# For recurrent policies, with PPO2, the number of environments run in parallel
# should be a multiple of nminibatches.
env = make_vec_env('fishing-v10', n_envs=4)

model = PPO2('MlpLstmPolicy', env, nminibatches=1, verbose=0)
model.learn(3000)

df = env.env_method("simulate", model, reps=10)
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
## simulate and plot results for reference
env.plot(df, "results/lstm.png")
