library(tidyverse)
library(lvplot)
library(arrow)
library(colorspace)
library(showtext)
library(patchwork)

source("dev/utils.R")

font_add("TeXGyreHeros", "~/.fonts/Tex-Gyre-Heros/texgyreheros-regular.otf")
showtext_auto()

dat_pth <- list.files(
  path = "dat/interim/06_postprocessing",
  pattern = "df_[a-zA-Z0-9]+_t1.feather",
  full.names = TRUE
)

lut_reg_col <- tribble(
  ~id, ~cluster, ~colors,
  -1, -1, rgb(0.4, 0.4, 0.4),
  0, 5, rgb(0.33, 0.66, 0.40),
  1, 1, rgb(0.77, 0.31, 0.32),
  2, 6, rgb(0.27, 0.42, 0.65),
  3, 3, rgb(0.36, 0.31, 0.56),
  4, 7, rgb(0.39, 0.71, 0.81),
  5, 4, rgb(0.80, 0.73, 0.46),
  6, 2, rgb(0.67, 0.61, 0.87)
)

nam <- str_extract(basename(dat_pth), "df_([A-Za-z]+)", group = 1)

clusters <- read_feather("dat/interim/06_postprocessing/df_cluster_out_2792936.feather") |>
  select(x, y, id = umap_hdbscan) |>
  drop_na()

progenitors <- map(dat_pth, read_feather) |>
  reduce(left_join, by = c("x", "y")) |>
  left_join(clusters, by = c("x", "y")) |>
  drop_na() |>
  mutate(SR = SR * 100) |>
  mutate(HS = if_else(HS > 5, NA, HS * 100)) |>
  mutate(SWE = if_else(SWE > 1000, NA, SWE)) |>
  mutate(RR = RR * 365) |>
  select(-x, -y) |>
  pivot_longer(dem:T, names_to = "progenitor") |>
  left_join(lut_reg_col, by = "id") |>
  mutate(cluster = as.factor(cluster)) |>
  mutate(progenitor = gsub("^dem$", "elevation [m]", progenitor)) |>
  mutate(progenitor = gsub("^ET0$", "reference evapotranspiration [kg/m²]", progenitor)) |>
  mutate(progenitor = gsub("^RR$", "precipitation [mm]", progenitor)) |>
  mutate(progenitor = gsub("^T$", "temperature [°C]", progenitor)) |>
  mutate(progenitor = gsub("^HS$", "snow depth [cm]", progenitor)) |>
  mutate(progenitor = gsub("^SR$", "relative sunshine duration [%]", progenitor)) |>
  mutate(progenitor = gsub("^SWE$", "snow water equivalent [kg/m²]", progenitor))

minor_breaks <- rep(1:9, 21) * (10^rep(-10:10, each = 9))

p1 <- plot_var(data = progenitors, vars = "elevation [m]") +
  # scale_y_log10(
  #  name = "",
  #  breaks = rep(c(1:3, 5), 21) * 10^(-10:10),
  #  minor_breaks = minor_breaks,
  # ) +
  ylab("") +
  xlab("")

p2 <- plot_var(data = progenitors, vars = "precipitation [mm]") +
  # scale_y_log10(
  #  name = "",
  #  breaks = rep(c(1:3, 5), 21) * 10^(-10:10),
  #  minor_breaks = minor_breaks,
  # ) +
  ylab("") +
  xlab("") +
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  coord_flip(ylim = c(470, 2950))

p3 <- plot_var(data = progenitors, vars = "relative sunshine duration [%]") +
  ylab("")

p4 <- plot_var(data = progenitors, vars = "temperature [°C]") +
  xlab("") +
  ylab("") +
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

p5 <- plot_var(data = progenitors, vars = "reference evapotranspiration [kg/m²]") +
  xlab("") +
  ylab("")

p6 <- plot_var(data = progenitors, vars = "snow depth [cm]") +
  scale_y_log10(
    name = "",
    breaks = 10^(-10:10),
    minor_breaks = minor_breaks
  ) +
  xlab("") +
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  coord_flip(ylim = c(0.1, 400.5))

pp <- ((p1 | p2) / (p3 | p4) / (p5 | p6))

ggsave("plt/progenitor_distribution.png", pp, width = 180, height = 200, units = "mm", dpi = 300)
