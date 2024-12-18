# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# plot distribution of progenitor parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

library("tidyverse")
library("stars")
library("sf")
library("lvplot")
library("colorspace")
library("showtext")
library("patchwork")

source("dev/utils.R")

font_add("TeXGyreHeros", "~/.fonts/Tex-Gyre-Heros/texgyreheros-regular.otf")
showtext_auto()

convert_stars <- function(x) {
  x |>
    st_as_sf(as_points = TRUE) |>
    sfc_as_cols() |>
    st_drop_geometry() |>
    as_tibble()
}

uh_col <- tribble(
  ~cluster_uh, ~cluster, ~colors,
  -1, -1, rgb(0.4, 0.4, 0.4), # "#666666"
  1, 1, rgb(0.33, 0.66, 0.40), # "#54A866"
  2, 2, rgb(0.77, 0.31, 0.32), # "#C44F52"
  3, 3, rgb(0.27, 0.42, 0.65), # "#456BA6"
  4, 7, rgb(0.36, 0.31, 0.56), # "#5C4F8F"
  5, 5, rgb(0.39, 0.71, 0.81), # "#63B5CF"
  6, 6, rgb(0.80, 0.73, 0.46), # "#CCBA75"
  7, 4, rgb(0.67, 0.61, 0.87) # "#AB9CDE"
)

km_col <- tribble(
  ~cluster_km, ~cluster, ~colors,
  1, 1, "#999999",
  2, 2, "#009e73",
  3, 3, "#e69f00",
  4, 4, "#0072b2",
  5, 5, "#cc79a7",
  6, 6, "#d55e00",
  7, 7, "#56b4e9"
)

basepar <- read_ncdf("dat/interim/07_cluster_evaluation/base_variables.nc") |>
  convert_stars() |>
  mutate(RR = RR * 365)

clusters_uh <- read_stars("dat/out/physioclimatic_clusters_raster_AT.tif") |>
  convert_stars() |>
  rename(cluster_uh = physioclimatic_clusters_raster_AT.tif)

clusters_km <- read_stars("dat/out/pca_kmeans_clusters_raster_at.tif") |>
  convert_stars() |>
  rename(cluster_km = pca_kmeans_clusters_raster_at.tif)

progenitors <- basepar |>
  left_join(clusters_uh, by = join_by(x, y)) |>
  left_join(clusters_km, by = join_by(x, y)) |>
  select(-x, -y) |>
  pivot_longer(RR:slope, names_to = "progenitor") |>
  mutate(progenitor = gsub("^RR$", "precipitation [mm]", progenitor)) |>
  mutate(progenitor = gsub("^T$", "temperature [°C]", progenitor)) |>
  mutate(progenitor = gsub("^slope$", "slope [°]", progenitor))

progenitors |>
  filter(progenitor == "precipitation [mm]") |>
  group_by(cluster_uh) |>
  summarize(rr_max = max(value))

pp1 <- progenitors |>
  select(-cluster_km) |>
  left_join(uh_col, by = "cluster_uh") |>
  mutate(cluster = as.factor(cluster)) |>
  drop_na()

pp2 <- progenitors |>
  select(-cluster_uh) |>
  left_join(km_col, by = "cluster_km") |>
  mutate(cluster = as.factor(cluster)) |>
  drop_na()

rm(basepar, clusters_uh, clusters_km, progenitors)

p1 <- plot_var(data = pp1, vars = "slope [°]", colors = uh_col$colors) +
  scale_y_continuous(breaks = seq(0, 50, 10))
p2 <- plot_var(data = pp1, vars = "precipitation [mm]", colors = uh_col$colors) +
  scale_y_continuous(breaks = seq(0, 2500, 500), limits = c(450, 2900))
p3 <- plot_var(data = pp1, vars = "temperature [°C]", colors = uh_col$colors)

p4 <- plot_var(data = pp2, vars = "slope [°]", colors = km_col$colors) +
  scale_x_discrete(position = "top") +
  scale_y_continuous(breaks = seq(0, 50, 10))
p5 <- plot_var(data = pp2, vars = "precipitation [mm]", colors = km_col$colors) +
  scale_x_discrete(position = "top") +
  scale_y_continuous(breaks = seq(0, 2500, 500), limits = c(450, 2900))
p6 <- plot_var(data = pp2, vars = "temperature [°C]", colors = km_col$colors) +
  scale_x_discrete(position = "top")

pp <- ((p1 | p4) / (p2 | p5) / (p3 | p6))

ggsave("plt/progenitor_distribution.png", pp, width = 180, height = 200, units = "mm", dpi = 300)
