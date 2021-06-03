#!/bin/bash

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py \
  --algo sac --env fishing-v1 -n 600000 -optimize \
  --n-trials 40 --n-jobs 1 --sampler tpe --pruner median \
  --storage sqlite:///tuning.db \
  --study-name sac-fishingv1

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py \
  --algo td3 --env fishing-v1 -n 300000 -optimize \
  --n-trials 20 --n-jobs 1 --sampler tpe --pruner median \
  --storage sqlite:///tuning.db \
  --study-name td3-fishingv1
  
CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo a2c --env fishing-v1 -n 300000 -optimize \
  --n-trials 200 --storage sqlite:///tuning.db   --study-name a2c-fishingv1

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo ppo --env fishing-v1 -n 300000 -optimize  \
  --n-trials 20 --storage sqlite:///tuning.db   --study-name ppo-fishing-v1

CUDA_VISIBLE_DEVICES="1" python ../tuning/zoo_train.py   --algo ddpg --env fishing-v1 -n 300000 -optimize \
  --n-trials 100 --n-jobs 4 --sampler tpe --pruner median   \
  --storage sqlite:///tuning.db   --study-name ddpg-fishingv1
  