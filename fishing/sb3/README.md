## Notes

When running `sb3_sac.py`, run `pip3 install -r requirements.txt` in a separate virtual environment. 
The following are general descriptions of the files found here.

- `sb3_sac.py`: This trains an agent using SAC from stable_baselines3 to interact with fishing-v1 i.e. the continous action space fishing environment. The hyperparameters here are not leading to consistent near-optimal performance just yet. I recently changed the episode length on the base environment, which degraded performance here even more.

- `tuning_sb3_sac.py`: This is the script that executes a round of tuning.

- `tuning_sb3_utils.py`: This script contains all the workings to carry out the tuning round that is called by `tuning_sb3_sac.py`. Of note here are the test distributions of hyperparameter values that you can adjust. This section is found at the bottom of the script.

- `callbacks_sb3.py`: This also contains some objects that are used when tuning, but nothing that I have edited from rl-zoo. 

- `plot.py`: This creates an .png of an averaged trajectory from an agent over 100 episodes. 

- `plot2.py`: This creates a .png that shows state, action and reward dynamics for multiple replicates.

- `train.py`: This is a general training script that allows the user to train an agent with a range of options via command-line arguments. E.g. `python train.py --algo sac --n_timesteps 200000` will train an agent using the hyperparameters found in `hyperparams/sac.yml`. Checkout the script to see other available CL arguments. 

