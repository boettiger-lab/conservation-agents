import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os

#file = os.path.basename(__file__)
file = "multi_env.py"
url = hash_url(file) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/leaderboard"


ENV = "fishing-v1"    
env = gym.make(ENV)


## A2C ######################################################################

# FIXME load best tuned parameters first...
# Trial 328 finished with value: 8.025644302368164 and parameters: {'gamma': 0.98, 'normalize_advantage': False, 'max_grad_norm': 0.3, 'use_rms_prop': True, 'gae_lambda': 0.98, 'n_steps': 16, 'lr_schedule': 'linear', 'lr': 0.03490204662520112, 'ent_coef': 0.00026525398345043097, 'vf_coef': 0.18060066335808234, 'log_std_init': -1.1353269076856574, 'ortho_init': True, 'net_arch': 'medium', 'activation_fn': 'relu'}. 

model = A2C('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("A2C", ENV, mean_reward, std_reward, url)
print("algo:", "A2C", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/a2c.png")
#policy = env.policyfn(model, reps=10)
#env.plot(policy, "results/a2c-policy.png")


