library("tidyverse")
library("arrow")
library("ranger")
library("mlr3")
library("mlr3learners")
library("mlr3tuning")
library("mlr3mbo")

get_n <- function(x) length(unique(x$umap_hdbscan)) - 1

sfc_as_cols <- function(x, geometry, names = c("x", "y")) {
  if (missing(geometry)) {
    geometry <- sf::st_geometry(x)
  } else {
    geometry <- rlang::eval_tidy(enquo(geometry), x)
  }
  stopifnot(inherits(x, "sf") && inherits(geometry, "sfc_POINT"))
  ret <- sf::st_coordinates(geometry)
  ret <- tibble::as_tibble(ret)
  stopifnot(length(names) == ncol(ret))
  x <- x[, !names(x) %in% names]
  ret <- setNames(ret, names)
  dplyr::bind_cols(x, ret)
}

read_srd <- function(feather_file) {
  feather_file |>
    read_feather() |>
    drop_na() |>
    select(-x, -y) |>
    mutate(
      umap_hdbscan = as.factor(umap_hdbscan),
      dtm_geomorphons_mode = as.factor(dtm_geomorphons_mode)
    ) |>
    rename_all(.funs = list(~ stringr::str_replace(., "\\-", "_")))
}

random_forest <- function(srd_data, id = "srd", resampling_strategy = rsmp("cv", folds = 5)) {
  # Setup classification task
  task <- as_task_classif(srd_data, target = "umap_hdbscan", id = id)

  # Define learner and search space
  learner <- lrn("classif.ranger",
    num.trees = 1000,
    mtry = to_tune(1, length(task$feature_names)),
    min.node.size = to_tune(p_int(1, 10)),
    sample.fraction = to_tune(0.2, 0.9),
    importance = "permutation",
    num.threads = 64L
  )

  # Setup tuning w/ mbo
  instance <- tune(
    tuner = tnr("mbo"),
    # https://mlr3mbo.mlr-org.com/reference/mbo_defaults.html
    task = task,
    learner = learner,
    resampling = resampling_strategy,
    measure = msr("classif.ce"),
    terminator = trm("evals", n_evals = 50)
  )

  # Set optimal hyperparameter configuration to learner
  learner$param_set$values <- instance$result_learner_param_vals

  # Train the learner on the full data set
  learner$train(task)

  return(learner)
}

nested_resampling <- function(srd_data, id = "srd", outer_resampling = rsmp("cv", folds = 5), inner_resampling = rsmp("cv", folds = 4)) {
  # Setup classification task
  task <- as_task_classif(srd_data, target = "umap_hdbscan", id = id)

  # Define learner and search space
  learner <- lrn("classif.ranger",
    num.trees = 1000,
    mtry = to_tune(1, length(task$feature_names)),
    min.node.size = to_tune(p_int(1, 10)),
    sample.fraction = to_tune(0.2, 0.9),
    importance = "permutation",
    num.threads = 64L
  )

  # Setup tuning w/ mbo
  at <- auto_tuner(
    tuner = tnr("mbo"),
    # https://mlr3mbo.mlr-org.com/reference/mbo_defaults.html
    learner = learner,
    resampling = inner_resampling,
    measure = msr("classif.ce"),
    terminator = trm("evals", n_evals = 20)
  )

  rr <- resample(task, at, outer_resampling, store_models = TRUE)

  return(rr)
}

get_score <- function(x) {
  x$score() |>
    select(classif.ce)
}

get_inner_tuning <- function(x) {
  x |>
    extract_inner_tuning_results() |>
    select(iteration:classif.ce)
}

get_importance <- function(ranger_model) {
  ranger_model$importance() |>
    tibble::enframe(.) |>
    rename(index = name, importance = value) |>
    mutate(type = if_else(grepl("^dtm_", index), "geomorphometry", as.character(NA))) |>
    mutate(type = if_else(grepl("^SPEI", index), "combined", type)) |>
    mutate(type = if_else(grepl("^ET0", index), "combined", type)) |>
    mutate(type = if_else(grepl("^PRCPTOT$", index), "precipitation", type)) |>
    mutate(type = if_else(grepl("^CDD$", index), "precipitation", type)) |>
    mutate(type = if_else(grepl("^T[NXy]", index), "temperature", type)) |>
    mutate(type = if_else(grepl("^GSL", index), "temperature", type)) |>
    mutate(type = if_else(grepl("^SU", index), "temperature", type)) |>
    mutate(type = if_else(grepl("^bio[0-9+]", index), "temperature", type)) |>
    mutate(type = if_else(grepl("^ETR_[0-9]+", index), "temperature", type)) |>
    mutate(type = if_else(grepl("TR$", index), "temperature", type)) |>
    mutate(type = if_else(grepl("^snow", index), "snow", type)) |>
    mutate(type = if_else(grepl("^swe", index), "snow", type)) |>
    mutate(type = if_else(grepl("^SR", index), "radiation", type)) |>
    left_join(lut_names, by = c("index" = "index_orig_name")) |>
    rename(index_name = index_nice_name) |>
    arrange(-importance) |>
    mutate(index_name = fct_reorder(index_name, -desc(importance)))
}

compute_bpstats <- function(x) {
  stats <- boxplot.stats(x)$stats
  out <- tibble(
    name = c("whisker_min", "lower_hinge", "median", "upper_hinge", "whisker_max"),
    value = stats
  ) |>
    pivot_wider()
  return(out)
}

theme_srd <- function() {
  ggplot2::theme_linedraw() +
    ggplot2::theme(
      text = element_text(
        family = "TeXGyreHeros",
        colour = "black",
        size = 30
      ),
    )
}

plot_var <- function(data, vars, colors) {
  df <- data |>
    filter(progenitor %in% vars)
  p <- ggplot(df, aes(x = cluster, y = value, color = cluster, fill = cluster)) +
    geom_boxplot(alpha = 0.8, show.legend = FALSE, outlier.shape = NA) +
    scale_fill_manual(values = colors) +
    scale_color_manual(values = colors) +
    facet_wrap(~progenitor) +
    theme_srd() +
    coord_flip() +
    ylab("") +
    xlab("")
  return(p)
}
