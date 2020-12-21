#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
python ../tuning/zoo_train.py   --algo a2c --env conservation-v3 -n 200000 -optimize \
  --n-trials 40 --n-jobs 4 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name a2c-cons-v3 &

python ../tuning/zoo_train.py   --algo ppo --env conservation-v3 -n 200000 -optimize \
  --n-trials 40 --n-jobs 4 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name ppo-cons-v3 &

 python ../tuning/zoo_train.py   --algo td3 --env conservation-v3 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name td3-cons-v3 &

python ../tuning/zoo_train.py   --algo ddpg --env conservation-v3 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name ddpg-cons-v3 &

python ../tuning/zoo_train.py   --algo sac --env conservation-v3 -n 200000 -optimize \
  --n-trials 10 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name sac-cons-v3 &
  
  
  