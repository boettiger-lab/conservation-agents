import gym
import gym_fishing
from gym_fishing.envs.shared_env import plot_mdp, simulate_mdp
from stable_baselines3 import SAC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="online_sac_tuned",
    help="Model name",
)
args = parser.parse_args()

env = gym.make("fishing-v1")
model = SAC.load(args.model)
eval_df = simulate_mdp(env, model, 10)
plot_mdp(env, eval_df, output=f"trash_{args.model}.png")
