import argparse
import sys
sys.path.append("tuning")
from parse_hyperparameters import load_results

import gym_conservation


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
    parser.add_argument("-r", "--results", help="results folder", type=str, default="results")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False)
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=100000, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=0, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=None)

    args = parser.parse_args()
    load_results(args.algo, 
                 args.env, 
                 log_dir = args.folder, 
                 seed = args.seed, 
                 verbose = args.verbose, 
                 outdir = args.results)


if __name__ == "__main__":
    main()

