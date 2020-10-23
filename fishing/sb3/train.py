import gym
import gym_fishing
import numpy as np
import torch as th
import yaml
import argparse

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from utils import StoreDict, ALGOS

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", default="sac", type=str,
                        help="Specify which of the algorithms to use")
    parser.add_argument("-s", "--seed", default=0, type=int,
                        help="Select the seed")
    parser.add_argument("-v", "--verbose", default=2, type=int,
                        help="Verbosity option when training")
    parser.add_argument("-n", "--n_timesteps", default=int(3e5), type=int,
                        help="Number of timesteps used for training")
    parser.add_argument("-e", "--env", default="fishing-v1", type=str,
                        help="Select which environment to use")
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, default={},
           help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("-l", "--log", default=1000, type=int,
                        help="Specify the logging interval when trainig")
    parser.add_argument("--fn", default="sac", type=str,
                        help="Select a filename to save the model to")
    args = parser.parse_args()

    with open(f"hyperparams/{args.algo}.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f) 
        hyperparams = hyperparams_dict[args.env]

    if "policy_kwargs" in hyperparams.keys():
        # Convert to python object if needed
        if isinstance(hyperparams["policy_kwargs"], str):
            hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    

    env = gym.make(args.env, **args.env_kwargs)

    model = ALGOS[args.algo](
                "MlpPolicy", 
                env=env, 
                seed=args.seed, 
                verbose=args.verbose, 
                **hyperparams)

    try:
        model.learn(args.n_timesteps, log_interval=args.log)
        #model.save("models/" + args.fn)
    except KeyboardInterrupt:
        pass
    finally:
        # Release resources
        env.close()


