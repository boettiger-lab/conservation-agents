

## R dependencies
library(reticulate)
library(tidyverse)
library(future.apply)


## We do this in parallel sessions on the GPU
#Sys.setenv("CUDA_VISIBLE_DEVICES" = "0")
#plan("multisession")

##  Turn CUDA off. then no need for parallelization either
Sys.setenv("CUDA_VISIBLE_DEVICES" = "")
plan("sequential")


## NB: Most integers must be explicitly typed as such (e.g. `1L` for `1`)

result_path <- function(prefix, i, dir = "results"){
  file.path(dir, paste0(prefix, "_", algo, "_", i, ".csv"))
}

## self-contained iterable, for parallel execution
train <- function(i = 1, 
                  seed = i,
                  ENV = "fishing-v1", 
                  algo = "TD3", 
                  steps = 300000L, 
                  tensorboard_log="/var/log/tensorboard",
                  dir = "results", 
                  ...){
  # Load python modules
  gym_fishing <- import("gym_fishing")
  gym         <- import("gym")
  sb3         <- import ("stable_baselines3")

  env <- gym$make(ENV, ...)
  init_model <- sb3[[algo]]
  model <- init_model('MlpPolicy', 
                      env,
                      seed = seed,
                      verbose = 0L,
                      tensorboard_log=tensorboard_log)
  model$learn(total_timesteps=as.integer(steps))

  ## save models
  dir.create("results", FALSE, TRUE)
  model$save(file.path(dir, paste0(algo, i)))

  
  
  sims <- env$simulate(model, reps=50L)
  policy <- env$policyfn(model, reps=500L)
  
  ## Save simulations and policyfn
  readr::write_csv(sims, result_path("sims", i, dir))
  readr::write_csv(policy, result_path("policy", i, dir))
  
  list(sims, policy)
}

####### Here we go.  Sit tight, this is gonna take a while! ################

algo = "SAC"
res <- future.apply::future_lapply(1:5, 
                                   train, 
                                   ENV = "fishing-v1", 
                                   steps = 300000L, 
                                   algo = algo,
                                   sigma = 0.05)
  
###################################################




## Load results
sims <- map_dfr(result_path("sims", 1:5), read_csv, .id = "model")
policy <-  map_dfr(result_path("policy", 1:5), read_csv, .id = "model")

## Some fun plotting

low <- function(x) quantile(x, probs = seq(0,1,by=0.05))[2]
high <- function(x) quantile(x, probs = seq(0,1,by=0.05))[20]

## Note no variation within models
policy %>% 
  group_by(state, model) %>% 
  summarise(mean_action = mean(action), 
            low = low(action), 
            high = high(action))  %>%
  ggplot(aes(state, mean_action)) + 
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2) + facet_wrap(~model)

p2 <- policy %>% 
  group_by(state) %>% 
  summarise(mean_action = mean(action), 
            low = low(action), 
            high = high(action)) 
p2 %>%
  ggplot(aes(state, mean_action)) + geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2)

policy %>% ggplot(aes(state, action, col=model)) + geom_line()
ggsave(paste0("results/",algo,"-policy.png"))




sims %>% group_by(model, rep) %>% 
  mutate(reward = cumsum(reward)) %>% 
  ungroup() %>%
  pivot_longer(cols = c(state, action, reward)) %>%
  ggplot(aes(time, value, col = rep)) + 
  geom_line(alpha=.8) +
  facet_grid(name~model, scales = "free")


ggsave(paste0("results/", algo, "-sims.png"))



