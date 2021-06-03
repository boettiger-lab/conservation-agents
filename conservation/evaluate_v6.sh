#!/bin/bash

CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo a2c --env conservation-v6 -n 10000000 --n-envs 16 -tb /var/log/tensorboard/eval_8M &
CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo ppo --env conservation-v6 -n 1000000 --n-envs 16 -tb /var/log/tensorboard/eval_8M &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo ddpg --env conservation-v6 -n 8000000 -tb /var/log/tensorboard/eval_8M  &
CUDA_VISIBLE_DEVICES="0" python ../tuning/evaluate_tuned.py --algo td3 --env conservation-v6 -n 6000000  -tb /var/log/tensorboard/eval_6M  &
CUDA_VISIBLE_DEVICES="1" python ../tuning/evaluate_tuned.py --algo sac --env conservation-v6 -n 8000000  -tb /var/log/tensorboard/eval_8M  &

