# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# compute permutation feature importance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

source("dev/utils.R")

dat_pth <- list.files(
  path = "dat/interim/06_postprocessing/",
  pattern = "df_cluster_out_[0-9]+.feather",
  full.names = TRUE
)
ids <- parse_number(basename(dat_pth))

# load data
srd <- lapply(dat_pth, read_srd)

# tune random forest w/ mlr3mbo
srd_rf <- lapply(srd, random_forest)
names(srd_rf) <- ids
saveRDS(srd_rf, "dat/interim/06_postprocessing/ranger_mbo.rds")

# estimate performance with nested resampling
srd_rr <- lapply(srd, nested_resampling)
names(srd_rr) <- ids
saveRDS(srd_rr, "dat/interim/06_postprocessing/ranger_nested_resampling.rds")

lapply(srd_rr, get_score) |>
  bind_rows(.id = "key") |>
  summarize(
    min_ce = min(classif.ce),
    mean_ce = mean(classif.ce),
    max_ce = max(classif.ce)
  )

lapply(srd_rr, get_inner_tuning) |>
  bind_rows(.id = "key")
