#!/bin/bash

python ../tuning/load_tuned.py --algo a2c --env conservation-v5  &
python ../tuning/load_tuned.py --algo ppo --env conservation-v5   &
python ../tuning/load_tuned.py --algo ddpg --env conservation-v5 &
python ../tuning/load_tuned.py --algo td3 --env conservation-v5 &
python ../tuning/load_tuned.py --algo sac --env conservation-v5  &

