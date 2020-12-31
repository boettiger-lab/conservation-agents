
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


* hyperparameter extraction failing: PPO, TD3
* successful hyperpar extraction: SAC, DDPG, A2C

--------

Methods attempted to run with saved hyperparameters:


- Custom parser, in parse_hyperparameters.py:
  - function `best_hyperpars()` seems to successfully extract best hyperparameters from logs/ALGO/report_ENV-*.csv
  - helper functions format shorthand notations (e.g. `net_arch: "medium"`) into specifications for network kwargs,
    `action_noise` (TD3/DDPG), and learning_rate (if it's a function, `a2c` only)

Other methods: 


### CLI 

Passing `--hyperparams` cli argument to zoo's train.py script.  (overrides the yaml settings. NOTE: presumably uses the same notation tricks for function arguments as noted there).


## YAML

Editing `hyperparams/<model>.yml` entry for given environment, then running zoo's `train.py` (without optimization).  Note: creates a best_model.zip in logs, which does not get created when running in tuning mode (`--optimize`).  

- Notes: most hyperparameters are given directly as argument names.  Some tricks are required when argument takes a more complex object type,
  e.g. policy_kwargs is a literal string interpreted as code, "nn.ReLU" for activation function, while linear learning rate function uses a short-hand.
  
  like this (below). But note this still fails to reproduce the tuning values of the revised A2C tuning...

```
conservation-v5:
  n_envs: 4
  n_timesteps: !!float 3e5
  policy: 'MlpPolicy'
  use_sde: True
  ent_coef: 3.0001446473553074e-07
  gae_lambda: 0.8
  gamma: 0.995
  learning_rate: lin_0.012486195510232303
  max_grad_norm: 0.7
  n_steps: 128
  normalize_advantage: False
  use_rms_prop: False
  vf_coef: 0.20033289618388683
  policy_kwargs: "dict(log_std_init=-2.4585859800053926, net_arch=[dict(pi=[256, 256], vf=[256, 256])], ortho_init=False, activation_fn=nn.ReLU)"

## used to determine learning_rate, see above
#  lr: 0.012486195510232303
#  lr_schedule: 'linear'

## used to determine policy_kwargs
#  log_std_init: -2.4585859800053926
#  activation_fn: 'relu'
#  net_arch: 'medium'
#  ortho_init: False
  
## Check td3/ddpg examples  to see how action_noise is specified...
```
  


