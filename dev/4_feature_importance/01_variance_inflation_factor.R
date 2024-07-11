# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# compute variance inflation factor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

library("glue")
library("parallel")
library("car")

source("dev/utils.R")

create_formula <- function(target, features) {
  expl <- grep(pattern = target, x = features, value = TRUE, invert = TRUE) |>
    paste(., collapse = " + ")
  formula(glue::glue("{target} ~ {expl}"))
}

compute_vif <- function(x) {
  x |>
    vif(type = "predictor") |>
    as.data.frame() |>
    rownames_to_column(var = "feature") |>
    as_tibble() |>
    select(feature, matches("GVIF"))
}

dat_pth <- list.files(
  path = "dat/interim/06_postprocessing/",
  pattern = "df_cluster_out_[0-9]+.feather",
  full.names = TRUE
)

# load data
srd <- read_srd(dat_pth[[1]]) |>
  select(-umap_hdbscan)

cn <- colnames(srd)

formulas <- sapply(cn, create_formula, features = cn)

lms <- mclapply(formulas, lm, data = srd, mc.cores = 16L)

res <- mclapply(lms, compute_vif, mc.cores = 16L) |>
  bind_rows(.id = "target_var") |>
  filter(target_var != "dtm_geomorphons_mode") |>
  rename(GVIF_normalized = `GVIF^(1/(2*Df))`)

out <- res |>
  group_by(feature) |>
  summarize(
    gvif = mean(GVIF, na.rm = TRUE),
    gvif_norm = mean(GVIF_normalized, na.rm = TRUE)
  ) |>
  arrange(-gvif)
