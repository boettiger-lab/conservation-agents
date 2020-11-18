#!/bin/bash

python zoo_train.py \
  --algo sac --env fishing-v1 -n 300000 -optimize \
  --n-trials 1000 --n-jobs 3 --sampler random --pruner median \
  --tensorboard-log /var/log/tensorboard/train_gpu &

CUDA_VISIBLE_DEVICES="" python zoo_train.py \
  --algo sac --env fishing-v1 -n 300000 -optimize \
  --n-trials 1000 --n-jobs 3 --sampler random --pruner median \
  --tensorboard-log /var/log/tensorboard/train &