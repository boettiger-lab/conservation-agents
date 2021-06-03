#!/bin/bash

## Trains an algo using the tuned parameters
CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo a2c --env fishing-v1 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo ppo --env fishing-v1 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo ddpg --env fishing-v1 -n 300000 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo sac --env fishing-v1 -n 300000  -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo td3 --env fishing-v1 -n 300000  -tb /var/log/tensorboard/evaluate  &

