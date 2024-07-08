# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# polygonize subregion raster
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

library("terra")

# replace valley clusters
r <- rast("dat/interim/06_postprocessing/da_cluster_out_2792936.tif") |>
  subst(from = 4, to = 5) |>
  subst(from = 2, to = 3)
NAflag(r) <- -1L

writeRaster(r, "dat/interim/06_postprocessing/main_clusters.tif",
  overwrite = TRUE, gdal = c("COMPRESS=NONE", "TFW=YES"), datatype = "INT4S"
)

# filter and polygonize
system("zsh dev/r.neighbors.sh 9")
gdal_polygonize <- paste(
  "gdal_polygonize.py -8 dat/interim/06_postprocessing/subregions_mode_9.tif",
  "-b 1 -f 'GPKG' dat/interim/06_postprocessing/subregions_mode_9.gpkg subregions_mode_9 cluster"
)
system(gdal_polygonize)
