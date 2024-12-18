"""Postprocess cluster output into dataframe for feature importance analysis"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

keys = [
    "2792936",  # 7 cluster
    "50143028",  # 4 cluster
]

davars = xr.load_dataarray(
    "dat/interim/03_feature_definition/core_variables_mean_t1_r0.8.nc"
)
for key in keys:
    out_path = f"dat/interim/06_postprocessing/df_cluster_out_{key}.feather"
    out_path_gt = f"dat/interim/06_postprocessing/da_cluster_out_{key}.tif"
    for file in Path(
        "dat/interim/05_hyperparametertuning/post_bugfix/experiment_hdbscan_hyperparams"
    ).rglob(f"cluster_out_{key}.nc"):
        daclust = xr.load_dataarray(file)
        daclust = daclust.transpose("y", "x")
        daclust = daclust.rio.write_crs(3416)
        daclust.name = "geoclimatic clusters"
        daclust = daclust.where(~np.isnan(daclust.values), -9999).astype(int)

        # sort numbering by frequency
        new_order = (
            daclust.to_dataframe()["geoclimatic clusters"].value_counts().reset_index()
        )["index"]
        daclust_static = daclust.copy()
        cluster_ind = 1
        for _, val in new_order.items():
            if (val != -9999) and (val != -1):
                daclust = daclust.where(daclust_static != val, cluster_ind)
                cluster_ind += 1

        # write crs and save cluster
        daclust.name = "physioclimatic clusters"
        daclust = daclust.rio.write_nodata(-9999)
        daclust.rio.to_raster(out_path_gt, dtype=np.int16)
        if key == "2792936":  # save specific primary result#
            daclust.rio.to_raster(
                "dat/out/physioclimatic_clusters_raster_AT.tif", dtype=np.int16
            )
        dfvars = davars.to_dataframe().reset_index()
        dfclust = daclust.to_dataframe().reset_index()

        # some nan cleaning
        dfclust = dfclust.replace(-9999.0, np.NaN)
        dfvars = dfvars.dropna()
        dfclust = dfclust.dropna()
        dfraw = pd.concat([dfvars, dfclust], axis=0)

        dffinvars = dfvars.pivot(
            index=["x", "y"], values="climate_indicator", columns="variable"
        )
        dffinclust = dfclust.pivot(
            index=["x", "y"], values="physioclimatic clusters", columns="variable"
        )
        # save with feather
        dffin = pd.concat([dffinvars, dffinclust], axis=1).reset_index()
        dffin.to_feather(out_path)