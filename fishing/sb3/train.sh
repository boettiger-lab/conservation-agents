#!/bin/bash

python zoo_train.py \
  --algo sac --env fishing-v1 -n 300000 -optimize \
  --n-trials 1000 --n-jobs 3 --sampler random --pruner median \
  --storage sqlite:///tuning.db \
  --study-name sac-fishingv1


  
python zoo_train.py   --algo a2c --env fishing-v1 -n 300000 -optimize   --n-trials 400 --n-jobs 3 --sampler random --pruner median   --storage sqlite:///tuning.db   --study-name a2c-fishingv1

python zoo_train.py   --algo ppo --env fishing-v1 -n 300000 -optimize   --n-trials 100 --n-jobs 4 --sampler random --pruner median   --storage sqlite:///tuning.db   --study-name ppo-fishingv1