import gym
import gym_fishing
from gym_fishing.envs.shared_env import plot_mdp, simulate_mdp
from d3rlpy.algos import CQL
from d3rlpy.wrappers.sb3 import SB3Wrapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="offline_cql_from_tuned_sac.pt",
    help="Model name",
)
args = parser.parse_args()

env = gym.make("fishing-v1")

cql_agent = CQL()
cql_agent.build_with_env(env)
cql_agent.load_model(args.model)

wrapped_model = SB3Wrapper(cql_agent)

# Evaluating the agent
offline_agent_df = simulate_mdp(env, wrapped_model, 10)
plot_mdp(env, offline_agent_df, output=f"trash_{args.model[:-3]}.png")
