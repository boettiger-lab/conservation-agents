import argparse
import sys
sys.path.append("tuning")
from parse_hyperparameters import train_from_logs

import gym_conservation


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("--eval_env", help="eval environment ID", type=str, default=None)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False)
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=100000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=4, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=0, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)
    parser.add_argument("-i", "--i", help="use i'th best model", default="0", type=int)

    args = parser.parse_args()
    train_from_logs(args.algo, 
                    args.env, 
                    eval_env = args.eval_env,
                    log_dir = args.folder, 
                    total_timesteps = args.n_timesteps,
                    tensorboard_log = args.tensorboard_log,
                    seed = args.seed, 
                    verbose = args.verbose, 
                    n_envs = args.n_envs,
                    outdir = "results",
                    use_sde = False,
                    i = args.i)


if __name__ == "__main__":
    main()

