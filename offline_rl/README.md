Sharing some initial poking around I did with offline RL, particularly the (D3RLPY)[https://d3rlpy.readthedocs.io/en/v0.91/] package.

In train_online_offline.py, a SAC agent is trained on fishing-v1. We then use this agent's replay buffer as the dataset to train an offline CQL agent. The hope is that the CQL agent will learn the same constant escapement-like policy of the SAC run. In my initial run however, this did not happen which is likely due to bad hyperparameters. We are probably going to have to spin-up a tuning script for D3RLPY.  

The eval scripts make standard plots of the SAC and CQL agent.

I use command line flags throughout these python scripts, so run `python *.py -h` to know what flags to use.
