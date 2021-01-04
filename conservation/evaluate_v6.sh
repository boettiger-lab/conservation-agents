#!/bin/bash

CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo a2c --env conservation-v6 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate &
CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo ppo --env conservation-v6 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo ddpg --env conservation-v6 -n 300000 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo td3 --env conservation-v6 -n 300000  -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo sac --env conservation-v6 -n 300000  -tb /var/log/tensorboard/evaluate  &

