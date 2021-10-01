import stable_baselines3 as sb3
from stable_baselines3 import SAC
import gym_fishing
import d3rlpy
from d3rlpy.algos import CQL
from d3rlpy.wrappers.sb3 import to_mdp_dataset
import yaml
import gym
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tuned-sac",
    dest="tuned_sac",
    action="store_true",
    default=False,
    help="Flag whether to use tuned or untuned SAC agent",
)
args = parser.parse_args()


def main():
    # Reading hyperparameters from yaml file
    with open(f"hyperparams/sac_tuned.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    env_name = "fishing-v1"
    env = gym.make(env_name)

    # Getting yaml in right format to input to SB3
    net_arch = {
        "small": dict(pi=[64, 64], qf=[64, 64]),
        "med": dict(pi=[256, 256], qf=[256, 256]),
        "large": dict(pi=[400, 400], qf=[400, 400]),
    }[params["net_arch"]]

    policy_kwargs = dict(
        net_arch=net_arch, log_std_init=params["log_std_init"]
    )
    n_timesteps = 250000
    keys_to_remove = ["log_std_init", "net_arch"]
    [params.pop(key) for key in keys_to_remove]

    # Training with SB3
    model_name = "online_sac"
    if args.tuned_sac:
        model = SAC(
            "MlpPolicy", env, verbose=2, policy_kwargs=policy_kwargs, **params
        )
        model_name += "_tuned"
    else:
        model = SAC("MlpPolicy", env, verbose=2)
        model_name += "_untuned"
    # If model already exists, save it. Load otherwise.
    if os.path.isfile(f"{model_name}.zip"):
        del model
        model = SAC.load(model_name)
        model.load_replay_buffer(model_name)
    else:
        model.learn(total_timesteps=n_timesteps, log_interval=1000)
        model.save(model_name)
        model.save_replay_buffer(model_name)
    
    # Converting replay buffer to d3rlpy MDPDataset and training offline agent
    dataset = to_mdp_dataset(model.replay_buffer)
    offline_model = CQL(
        use_gpu=True,
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
    )

    offline_model.fit(
        dataset.episodes,
        n_steps_per_epoch=1,
        n_steps=5,
        scorers={
            "environment": d3rlpy.metrics.evaluate_on_environment(env),
            "value_scale": d3rlpy.metrics.average_value_estimation_scorer,
        },
    )
    offline_model.save_model(f"offline_cql_from_{model_name}.pt")


if __name__ == "__main__":
    main()
