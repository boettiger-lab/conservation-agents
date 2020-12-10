#!/bin/bash

python ../tuning/zoo_train.py   --algo a2c --env conservation-v2 -n 3000000 -optimize \
  --n-trials 500 --n-jobs 4 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name a2c-cons-v2


python ../tuning/zoo_train.py   --algo td3 --env conservation-v2 -n 1000000 -optimize \
  --n-trials 500 --n-jobs 1 --sampler random --pruner median \
  --storage sqlite:///tuning.db   --study-name td3-cons-v2

