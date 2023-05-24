# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Check region aggregates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# packages
library(tidyverse)
library(arrow)

fl <- list.files("dat/interim/06_postprocessing", pattern = "*.feather", full.names = TRUE)

# "dat/interim/06_postprocessing/df_cluster_out_2067064.feather"
read_feather(fl[2]) %>%
  as_tibble() %>%
  select(-x, -y, -umap_hdbscan) %>%
  drop_na() %>%
  summarize_all(.funs = list(min, mean, max)) %>%
  pivot_longer(cols = everything()) %>%
  separate(col = name, into = c("name", "fun"), sep = "_fn") %>%
  mutate(fun = gsub("1", "min", fun)) %>%
  mutate(fun = gsub("2", "mean", fun)) %>%
  mutate(fun = gsub("3", "max", fun)) %>%
  pivot_wider(names_from = fun, values_from = value) %>%
  print(n = 36)
