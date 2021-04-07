#!/bin/bash


CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo a2c --env Pendulum-v0 -n 300000 -optimize \
  --n-trials 60 --n-jobs 4 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db --study-name a2c-pendulum-v0 &

CUDA_VISIBLE_DEVICES="0" python ../tuning/zoo_train.py   --algo ppo --env Pendulum-v0 -n 300000 -optimize \
  --n-trials 60 --n-jobs 4 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name ppo-pendulum-v0 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo td3 --env Pendulum-v0 -n 300000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name td3-pendulum-v0 &

### Consider different samplers for the algos that tune unsuccessfully.  tpe sampler?
CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo ddpg --env Pendulum-v0 -n 300000 -optimize \
  --n-trials 40 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name ddpg-pendulum-v0 &

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo sac --env Pendulum-v0 -n 300000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler skopt --pruner median \
  --storage sqlite:///tuning.db   --study-name sac-pendulum-v0 &
  
  
  