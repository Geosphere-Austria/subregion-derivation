library(sf)
library(dplyr)
library(tidyr)

compute_area_percentage <- function(x, digits = 4) {
  perc <- st_area(x) / st_area(st_union(x))
  res <- perc %>%
    units::drop_units() %>%
    round(digits = digits)
  return(res)
}

sort_perc <- function(x) {
  st_drop_geometry(x) %>%
    arrange(-perc) %>%
    print(n = nrow(.))
}

crsm <- read_sf("dat/raw/crsm/Shape_CRSM.shp") %>%
  st_transform(3416) %>%
  mutate(perc = compute_area_percentage(.)) %>%
  select(id = Id, perc)
sort_perc(crsm)

fgr <- read_sf("dat/raw/fgr/WLamPoly.shp") %>%
  st_transform(3416) %>%
  mutate(perc = compute_area_percentage(.)) %>%
  select(reg_id = Wuchsge1, reg_name = Wuchsnam, perc)
sort_perc(fgr)

fgr_main <- fgr %>%
  separate(reg_id, c("main", "sub")) %>%
  group_by(main) %>%
  summarize(geometry = st_union(geometry)) %>%
  mutate(perc = compute_area_percentage(.)) %>%
  select(main, perc)
sort_perc(fgr_main)
