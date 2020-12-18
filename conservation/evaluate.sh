#!/bin/bash

mkdir -p results/sac
mkdir -p results/td3
mkdir -p results/ddpg
mkdir -p results/a2c
mkdir -p results/ppo


CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo a2c --env conservation-v5 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate &
CUDA_VISIBLE_DEVICES="" python ../tuning/evaluate_tuned.py --algo ppo --env conservation-v5 -n 300000 --n-envs 4 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo ddpg --env conservation-v5 -n 300000 -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo sac --env conservation-v5 -n 300000  -tb /var/log/tensorboard/evaluate  &
python ../tuning/evaluate_tuned.py --algo td3 --env conservation-v5 -n 300000  -tb /var/log/tensorboard/evaluate  &

