import gym
import gym_fishing
from gym_fishing.models.policies import msy, escapement
from stable_baselines3.common.evaluation import evaluate_policy
from leaderboard import leaderboard, hash_url
import os


url = hash_url(os.path.basename(__file__)) # get hash URL at start of execution
tensorboard_log="/var/log/tensorboard/leaderboard"


ENV = "fishing-v1"    
env = gym.make(ENV)


## Constant Escapement ######################################################
model = escapement(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/escapement.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
leaderboard("ESC", ENV, mean_reward, std_reward, url)
print("algo:", "ESC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## MSY ######################################################################
model = msy(env)
df = env.simulate(model, reps=10)
env.plot(df, "results/msy.png")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
leaderboard("MSY", ENV, mean_reward, std_reward, url)
print("algo:", "MSY", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


# Consider running these in parallel?


## PPO ######################################################################

# load best tuned parameters...

model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("PPO", ENV, mean_reward, std_reward, url)
print("algo:", "PPO", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/ppo-policy.png")


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
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/a2c-policy.png")


## DDPG ######################################################################

# FIXME load best tuned parameters first...

model = DDPG('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("DDPG", ENV, mean_reward, std_reward, url)
print("algo:", "DDPG", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results for reference
df = env.simulate(model, reps=10)
env.plot(df, "results/ddpg.png")
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/ddpg-policy.png")

## SAC #######################################################################

# FIXME read from YAML
hyper = {'gamma': 0.99, 
         'lr': 1.8825727360507924e-05, 
         'batch_size': 512, 
         'buffer_size': 10000, 
         'learning_starts': 10000, 
         'train_freq': 1, 
         'tau': 0.005, 
         'log_std_init': -0.3072998266889968, 
         'net_arch': 'medium'}
policy_kwargs = dict(log_std_init=-3.67, net_arch=[256, 256])

model = SAC('MlpPolicy', 
            env, verbose=0, 
            use_sde=True,
            gamma = hyper['gamma'],
            learning_rate = hyper['lr'],
            batch_size = hyper['batch_size'],            
            buffer_size = hyper['buffer_size'],
            learning_starts = hyper['learning_starts'],
            train_freq = hyper['train_freq'],
            tau = hyper['tau'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)

## Evaluate model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard.leaderboard("SAC", ENV, mean_reward, std_reward, url)
print("algo:", "SAC", "env:", ENV, "mean reward:", mean_reward, "std:", std_reward)


## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/sac-tuned.png")


## TD3 ######################################################################

# load best tuned parameters...

model = TD3('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=300000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
leaderboard("TD3", ENV, mean_reward, std_reward, url)
print("algo:", "TD3", "env:",ENV, "mean reward:", mean_reward, "std:", std_reward)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/td3.png")
policy = env.policyfn(model, reps=10)
env.plot(policy, "results/td3-policy.png")

