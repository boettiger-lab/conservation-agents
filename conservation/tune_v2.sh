#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
python ../tuning/zoo_train.py   --algo a2c --env conservation-v2 -n 200000 -optimize \
  --n-trials 40 --n-jobs 4 --sampler random --pruner median \
  --storage sqlite:///tuning_v2.db   --study-name a2c-cons-v2 &

python ../tuning/zoo_train.py   --algo ppo --env conservation-v2 -n 200000 -optimize \
  --n-trials 40 --n-jobs 4 --sampler random --pruner median \
  --storage sqlite:///tuning_v2.db   --study-name ppo-cons-v2 &

 python ../tuning/zoo_train.py   --algo td3 --env conservation-v2 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning_v2.db   --study-name td3-cons-v2 &

python ../tuning/zoo_train.py   --algo ddpg --env conservation-v2 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning_v2.db   --study-name ddpg-cons-v2 &

python ../tuning/zoo_train.py   --algo sac --env conservation-v2 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning_v2.db   --study-name sac-cons-v2 &
  
  
  