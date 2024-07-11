"""script to process data needed for cluster evaluation"""

import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import json


def load_base_param_df(variable: str, par_dir: str = "dat/interim/06_postprocessing"):
    df_ = pd.read_feather(f"{par_dir}/df_{variable}_t1.feather")
    if "mean" in variable:
        variable = variable.rstrip("mean")
    print(f"..load {variable = }..")
    df_ = df_.astype({"x": int, "y": int, variable: float})
    df_["xy"] = df_.x.astype(str) + "-" + df_.y.astype(str)
    return df_


def load_base_param_tif(variable: str, par_dir: str = "dat/interim/06_postprocessing"):
    xda_ = rioxarray.open_rasterio(f"{par_dir}/da_{variable}_t1.tif")
    xda_ = xda_.expand_dims(
        variable=(
            [
                variable,
            ]
        )
    )
    if "mean" in variable:
        variable = variable.rstrip("mean")
    print(f"..load {variable = }..")
    return xda_.squeeze()


def load_input_features():
    xda_ = rioxarray.open_rasterio(
        "dat/out/physioclimatic_features_grid_AT_average_1992-2021.nc"
    )
    xda_ = xda_.rio.write_crs(3416)
    xda_ = xda_.drop(
        [
            "spatial_ref",
        ]
    )
    xda_ = xda_.transpose("variable", "y", "x")
    with open(
        "dat/interim/03_feature_definition/core_variable_groups_mean_t1_r0.8.json"
    ) as file:
        aidct = json.load(file)
    variables_dict = list(aidct.keys())
    return xda_.sel(variable=variables_dict)


# fname_in = "dat/interim/02_preprocessed_climate_normals/indicators_climate_normals.nc"
# fname_out = "dat/out/physioclimatic_features_grid_AT_average_1992-2021.nc"

clusters_tif = "dat/out/physioclimatic_clusters_raster_AT.tif"
clusters = rioxarray.open_rasterio(clusters_tif).squeeze()
clusters = clusters.where(clusters != -9999, np.nan)


features = load_input_features()
features = features.where(~clusters.isnull())
del features.attrs["NETCDF_DIM_EXTRA"]
del features.attrs["NETCDF_DIM_variable_DEF"]
del features.attrs["NETCDF_DIM_variable_VALUES"]
features.to_netcdf("dat/interim/07_cluster_evaluation/core_variables_r0.8.nc")

slope_ds = xr.load_dataset("dat/interim/01_concat_files/dtm_concat_indices.nc")
slope_da = slope_ds.sel(variable="dtm_slope_average")
slope_xda = slope_da["climate_indicator"]
slope_xda = slope_xda.rio.write_crs(3416)
slope_xda = slope_xda.rename("slope").drop(["variable", "spatial_ref"])
del slope_xda.attrs["grid_mapping"]
slope_xda.rio.to_raster("dat/interim/06_postprocessing/da_slope_t1.tif")
slope = slope_da.to_dataframe().reset_index()
slope = slope.drop("variable", axis=1)
slope = slope.rename({"climate_indicator": "slope"}, axis=1)
slope = slope.astype({"x": int, "y": int, "slope": float})
slope["xy"] = slope.x.astype(str) + "-" + slope.y.astype(str)
slope.to_feather("dat/interim/06_postprocessing/df_slope_t1.feather")
# base params in dataframe
RR_df = load_base_param_df("RRmean")
T_df = load_base_param_df("Tmean")
slope_df = load_base_param_df("slope")
# base params in xr.DataArray
RR_xda = load_base_param_tif("RRmean")
T_xda = load_base_param_tif("Tmean")
slope_xda = load_base_param_tif("slope")
slope_xda = slope_xda.where(slope_xda != -99999.0, np.nan)
base_vars = xr.concat([RR_xda, T_xda, slope_xda], dim="base_param")
base_vars = base_vars.assign_coords(dict(base_param=["RR", "T", "slope"])).drop(
    "variable"
)
# subselect cluster domain
base_vars = base_vars.where(~clusters.isnull())
base_vars.to_netcdf("dat/interim/07_cluster_evaluation/base_variables.nc")
base_vars.name = "value"

base_vars_df = (
    base_vars.to_dataframe()
    .drop(["band", "spatial_ref"], axis=1)
    .reset_index("base_param")
)
base_vars_df = base_vars_df.pivot(columns="base_param").dropna(how="all")
base_vars_df.columns = base_vars_df.columns.droplevel()
base_vars_df = base_vars_df.reset_index().astype(
    {"x": int, "y": int, "RR": float, "T": float, "slope": float}
)
base_vars_df.to_csv("dat/interim/07_cluster_evaluation/base_variables.csv")
