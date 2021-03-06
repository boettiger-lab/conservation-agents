#!/bin/bash


CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo a2c --env conservation-v5 -n 300000 -optimize \
  --n-trials 60 --n-jobs 4 --sampler tpe --pruner median \
  --study-name a2c-cons-v5 &

CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo ppo --env conservation-v5 -n 300000 -optimize \
  --n-trials 60 --n-jobs 4 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name ppo-cons-v5 &

CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo td3 --env conservation-v5 -n 300000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name td3-cons-v5 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo ddpg --env conservation-v5 -n 300000 -optimize \
  --n-trials 40 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name ddpg-cons-v5 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo sac --env conservation-v5 -n 300000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuningB.db   --study-name sac-cons-v5 &
  
  
  