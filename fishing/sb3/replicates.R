## R dependencies
library(reticulate)
library(tidyverse)

## NB: Most integers must be explicitly typed as such (e.g. `1L` for `1`)
##  Turn CUDA off for reproducibility, if necessary.
Sys.setenv("CUDA_VISIBLE_DEVICES" = "0")


result_path <- function(prefix, i, dir = "results"){
  file.path(dir, paste0(prefix, "_", algo, "_", i, ".csv"))
}

## self-contained iterable, for parallel execution
train <- function(i = 1, 
                  seed = i,
                  ENV = "fishing-v1", 
                  algo = "SAC", 
                  steps = 200000L, 
                  tensorboard_log="/var/log/tensorboard",
                  dir = "results", 
                  ...){
  # Load python modules
  gym_fishing <- import("gym_fishing")
  gym         <- import("gym")
  sb3         <- import ("stable_baselines3")
  torch       <- import("torch")
  np          <- import("numpy")
  
  torch$manual_seed(seed)
  np$random$seed(seed)
  
  env <- gym$make(ENV, ...)
  init_model <- sb3[[algo]]
  model <- init_model('MlpPolicy', 
                      env, 
                      verbose=0L,
                      tensorboard_log=tensorboard_log)
  model$learn(total_timesteps=as.integer(steps))

  ## save models
  dir.create("results")
  lapply(seq_along(models), function(i){
    models[[i]]$save(file.path(dir, paste0(algo, i)))
  })
  
  
  sims <- env$simulate(model, reps=50L)
  policy <- env$policyfn(model, reps=500L)
  
  ## Save simulations and policyfn
  readr::write_csv(sims, result_path("sims", i, dir))
  readr::write_csv(policy, result_path("policy", i, dir))
  
  list(sims, policy)
}

## Here we go.  Sit tight, this is gonna take a while!
## We do this in parallel sessions
library(future.apply)
plan("multisession")
res <- future.apply::future_lapply(1:5, train, sigma = 0.05)
  

## Load results
df <- vroom::vroom(result_path("sims", 1:5))
policy <- vroom::vroom(result_path("policy", 1:5))

## Some fun plotting

low <- function(x) quantile(x, probs = seq(0,1,by=0.05))[2]
high <- function(x) quantile(x, probs = seq(0,1,by=0.05))[20]

p2 <- policy %>% 
  group_by(state) %>% 
  summarise(mean_action = mean(action), 
            low = low(action), 
            high = high(action)) 
p2 %>%
  ggplot(aes(state, mean_action)) + 
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2)
ggsave("results/SAC-policy.png")

## repair NAs
as.na <- function(x){
  x[vapply(x, is.null, NA)] <- NA
  as.numeric(x)
}
sims <- df %>%
  as_tibble() %>%
  mutate(state = as.na(state),
         action = as.na(action)/100,
         reward = as.na(reward),
         rep = as.integer(rep))

sims %>% group_by(model, rep) %>% 
  mutate(reward = cumsum(reward)) %>% 
  ungroup() %>%
  pivot_longer(cols = c(state, action, reward)) %>%
  ggplot(aes(time, value, col = rep)) + 
  geom_line(alpha=.8) +
  facet_grid(name~model, scales = "free")


ggsave("results/SAC-sims.png")



