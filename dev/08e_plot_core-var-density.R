# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Plot empirical density of core variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# packages
library(tidyverse)
library(arrow)

# colors
region_col <- tibble(
  region = c(
    "Noise", "Austrian Alps", "Inneralpine Valleys", "Southeastern Alpine Foothills",
    "Southern Valleys", "Styria", "Southeastern Central Alps", "Eastern Lowlands",
    "Northeastern Basins", "Innviertel", "Granite and Gneis Plateau", "Danube"
  ),
  color = c(
    rgb(0.4, 0.4, 0.4),
    rgb(0.27, 0.42, 0.65),
    rgb(0.80, 0.73, 0.46),
    rgb(0.36, 0.31, 0.56),
    rgb(0.67, 0.61, 0.87),
    rgb(0.51, 0.45, 0.71),
    rgb(0.77, 0.31, 0.32),
    rgb(0.87, 0.52, 0.32),
    rgb(0.39, 0.71, 0.81),
    rgb(0.86, 0.55, 0.77),
    rgb(0.33, 0.66, 0.40),
    rgb(0.58, 0.47, 0.38)
  )
)


# lut
cluster_col <- tibble(
  cluster = -1L:9L,
  region = c(
    "Noise",
    "Austrian Alps",
    "Granite and Gneis Plateau",
    "Southeastern Central Alps",
    "Eastern Lowlands",
    "Inneralpine Valleys",
    "Southeastern Alpine Foothills",
    "Northeastern Basins",
    "Innviertel",
    "Danube",
    "Southern Valleys"
  )
) |>
  left_join(region_col, by = "region")

# feature names
lut_names <- read_csv("doc/lut_vars.csv") |>
  mutate(index_nice_name = gsub("variability", "var", index_nice_name))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# helper functions
q25 <- function(x) quantile(x, 0.25)
q75 <- function(x) quantile(x, 0.75)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

fl <- list.files(
  path = "dat/interim/06_postprocessing/",
  pattern = "df_cluster_out_[0-9]+.feather",
  full.names = TRUE
)

x <- read_feather(fl[2]) |>
  as_tibble() |>
  select(-x, -y) |>
  drop_na() |>
  group_by(umap_hdbscan)
# |>
#   summarize_all(list(min, q25, median, q75, max))

y <- x |>
  pivot_longer(cols = GSL:snow_depth_gt_1, names_to = "index_orig_name", values_to = "value") |>
  mutate(index_orig_name = gsub("-", "_", index_orig_name)) |>
  left_join(lut_names, by = "index_orig_name") |>
  mutate(cluster = factor(umap_hdbscan)) |>
  mutate(cluster = fct_relabel(cluster, ~ paste(cluster_col$region)))

p <- ggplot(y, aes(value, after_stat(density), color = cluster)) +
  geom_freqpoly() +
  scale_color_manual(values = cluster_col$color) +
  facet_wrap(~index_nice_name, scales = "free", ncol = 4) +
  theme_bw() +
  theme(
    text = element_text(
      family = "DejaVu",
      colour = "black",
      size = 16
    ),
    legend.position = "bottom"
  )

ggsave(filename = "plt/freqpoly_test.png", plot = p, width = 400, height = 480, units = "mm")
