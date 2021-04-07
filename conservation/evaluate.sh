#!/bin/bash

## Not robust to longer training, but at least most (except TD3) repeat or nearly repeat score achieved in tuning

CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo a2c --env conservation-v5 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate &
CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo ppo --env conservation-v5 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate  &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo ddpg --env conservation-v5 -n 300000 -tb /var/log/tensorboard/evaluate  &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo td3 --env conservation-v5 -n 300000  -tb /var/log/tensorboard/evaluate  &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo sac --env conservation-v5 -n 300000  -tb /var/log/tensorboard/evaluate  &

