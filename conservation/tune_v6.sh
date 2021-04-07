#!/bin/bash


CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo a2c --env conservation-v6 -n 3000000 -optimize \
  --n-trials 200 --n-jobs 4 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db --study-name a2c-cons-v6 &

CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo ppo --env conservation-v6 -n 3000000 -optimize \
  --n-trials 60 --n-jobs 4 --sampler skopt --pruner median \
  --storage sqlite:///tuningB.db   --study-name ppo-cons-v6 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo td3 --env conservation-v6 -n 3000000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuningB.db   --study-name td3-cons-v6 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo ddpg --env conservation-v6 -n 3000000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuningB.db   --study-name ddpg-cons-v6 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo sac --env conservation-v6 -n 3000000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name sac-cons-v6-b &
  
  
  