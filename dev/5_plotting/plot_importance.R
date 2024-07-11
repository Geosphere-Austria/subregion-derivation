# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# plot permutation feature importance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

library("colorspace")
library("showtext")

font_add("TeXGyreHeros", "~/.fonts/Tex-Gyre-Heros/texgyreheros-regular.otf")
showtext_auto()

source("dev/utils.R")

dat_pth <- list.files(
  path = "dat/interim/06_postprocessing/",
  pattern = "df_cluster_out_[0-9]+.feather",
  full.names = TRUE
)
ids <- parse_number(basename(dat_pth))
lut_names <- read_csv("doc/lut_vars.csv")

srd <- lapply(dat_pth, read_srd)
srd_rf <- read_rds("dat/interim/06_postprocessing/ranger_mbo.rds")
srd_imp <- lapply(srd_rf, get_importance)

lut_n_clust <- tibble(id = ids, n = sapply(srd, get_n))

# aggregated importance from models ----

imp <- srd_imp |>
  bind_rows(.id = "id") |>
  mutate(id = as.integer(id)) |>
  left_join(lut_n_clust) |>
  arrange(n, -importance) |>
  mutate(title = paste0("k = ", n, "(", id, ")")) |>
  mutate(title = fct_reorder(title, -desc(n)))

p <- ggplot(imp, aes(x = index_name, y = importance, color = type, alpha = importance)) +
  geom_point(size = 6) +
  coord_flip() +
  facet_wrap(~title, nrow = 1) +
  xlab("feature name") +
  ylab("permutation feature importance") +
  guides(color = guide_legend(title = "progenitor")) +
  scale_alpha(range = c(0.4, 1), guide = "none") +
  theme_linedraw() +
  theme(
    text = element_text(
      family = "DejaVu",
      colour = "black",
      size = 16
    ),
    legend.position = "bottom"
  )

ggsave("plt/importance.png", p, width = 400, height = 250, units = "mm")


# importance across different clustering results ----

imp_agg <- imp |>
  group_by(index_name) |>
  summarize(
    min = min(importance),
    mean = mean(importance),
    max = max(importance),
    type = first(type)
  ) |>
  arrange(-mean) |>
  mutate(index_name = fct_reorder(index_name, -desc(mean)))

p <- ggplot(imp_agg, aes(x = index_name, y = mean, color = type)) +
  geom_pointrange(aes(ymin = min, ymax = max)) +
  coord_flip() +
  theme_linedraw() +
  xlab("feature name") +
  ylab("permutation feature importance") +
  scale_color_discrete_qualitative("Dark3") +
  guides(color = guide_legend(title = "progenitor")) +
  theme(
    text = element_text(
      family = "DejaVu",
      colour = "black",
      size = 16
    )
  )

ggsave("plt/importance_aggregated.png", p, width = 300, height = 250, units = "mm")


# importance from nested resampling ----

srd_rr <- read_rds("dat/interim/06_postprocessing/ranger_nested_resampling.rds")
rr_imp <- lapply(srd_rr, \(x) map(x$learners, get_importance))

rr_7 <- rr_imp["2792936"] |>
  bind_rows(.id = "cv") |>
  mutate(cv = as.integer(cv)) |>
  group_by(index_name) |>
  summarize(
    min = min(importance),
    mean = mean(importance),
    max = max(importance),
    type = first(type)
  ) |>
  arrange(-mean) |>
  mutate(index_name = fct_reorder(index_name, -desc(mean)))

p <- ggplot(rr_7, aes(x = index_name, y = mean, color = type)) +
  geom_pointrange(aes(ymin = min, ymax = max)) +
  coord_flip() +
  xlab("feature name") +
  ylab("permutation feature importance") +
  scale_color_discrete_qualitative("Dark3") +
  guides(color = guide_legend(title = "progenitor")) +
  theme_linedraw() +
  theme(
    text = element_text(
      family = "TeXGyreHeros",
      colour = "black",
      size = 30
    ),
    legend.position = "bottom"
  )

ggsave("plt/importance_nested_resampling.png", p, width = 230, height = 150, units = "mm", dpi = 300)
