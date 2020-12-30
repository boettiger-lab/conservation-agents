
## PPO

- conservation-v5: fails.  446 (no action) vs tuned value 947
- fishing-v1

## A2C

- conservation-v5: nearly matching tuned (823 vs 831). (crashes only at end, but sometimes steady)
- fishing-v1: 

## DDPG

- conservation-v5: (tuning only gets 0-action so far, so does eval)
- fishing-v1: nearly optimal, at tuning value (7.75 vs 7.74). 

## TD3

not matching
- conservation-v5: fails; 0 action (~ 446 vs tuned at 855)
- fishing-v1: fails, harvest all (0.75 vs 7.75)

## SAC

- conservation-v5: nearly optimal, ~ 800
- fishing-v1: nearly optimal, close to tuning value (7.70 vs 7.75)

