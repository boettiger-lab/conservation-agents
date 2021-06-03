#!/bin/bash

## evaluate a saved agent on a given gym
#python ../tuning/enjoy.py --algo a2c --env fishing-v1 -f logs/ --exp-id 1 --load-best
#python ../tuning/enjoy.py --algo sac --env fishing-v1 -f logs/ --exp-id 2 --load-best

#python ../tuning/evaluate.py --algo a2c --env fishing-v1 --env-kwargs sigma:0.1 &
python ../tuning/evaluate_tuned.py --algo ppo --env fishing-v1 
#python ../tuning/evaluate_tuned.py --algo ddpg --env fishing-v1 &
python ../tuning/evaluate_tuned.py --algo sac --env fishing-v1
python ../tuning/evaluate_tuned.py --algo a2c --env fishing-v1

#python ../tuning/evaluate.py --algo td3 --env fishing-v1  &

