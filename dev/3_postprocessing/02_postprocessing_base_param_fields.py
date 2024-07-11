"""script to generate temperature, precipitation and elevation aggregates"""

import xarray as xr
import pandas as pd
import rioxarray

# load data
da_climdex = xr.open_dataarray("dat/interim/01_concat_files/climdex_concat_indices.nc")
da_snowgrid = xr.open_dataarray(
    "dat/interim/01_concat_files/misc_concat_snowgrid_indices.nc"
)
da_sparta_winfore = xr.open_dataarray(
    "dat/interim/01_concat_files/misc_concat_spartacus_winfore_indices.nc"
)

T = (
    da_climdex.sel(variable="Tyearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
RR = (
    da_climdex.sel(variable="RRyearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
SWE = (
    da_snowgrid.sel(variable="swe_tot_yearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
HS = (
    da_snowgrid.sel(variable="snow_depth_yearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
SR = (
    da_sparta_winfore.sel(variable="SR_yearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
ET0 = (
    da_sparta_winfore.sel(variable="ET0_yearmean")
    .sel(time=slice("01-01-1992", "01-01-2022"))
    .mean(dim="time")
)
# get the DEM from any SPARTACUS file
dem = xr.open_dataset("/dat/raw/grid/obs/spartacus_uncompressed/RRhr/RRhr1961.nc")[
    "dem"
]

del dem.attrs["grid_mapping"]
del dem.attrs["esri_pe_string"]

out_path = "dat/interim/06_postprocessing/"
out_path_T_df = f"{out_path}/df_Tmean_t1.feather"
out_path_T_da = f"{out_path}/da_Tmean_t1.tif"
out_path_RR_df = f"{out_path}/df_RRmean_t1.feather"
out_path_RR_da = f"{out_path}/da_RRmean_t1.tif"
out_path_dem_df = f"{out_path}/df_dem_t1.feather"
out_path_dem_da = f"{out_path}/da_dem_t1.tif"
out_path_SWE_df = f"{out_path}/df_SWEmean_t1.feather"
out_path_SWE_da = f"{out_path}/da_SWEmean_t1.tif"
out_path_HS_df = f"{out_path}/df_HSmean_t1.feather"
out_path_HS_da = f"{out_path}/da_HSmean_t1.tif"
out_path_SR_df = f"{out_path}/df_SRmean_t1.feather"
out_path_SR_da = f"{out_path}/da_SRmean_t1.tif"
out_path_ET0_df = f"{out_path}/df_ET0mean_t1.feather"
out_path_ET0_da = f"{out_path}/da_ET0mean_t1.tif"


# process data
def prep_da(da_in: xr.DataArray) -> xr.DataArray:
    da_in = da_in.transpose("y", "x")
    return da_in.rio.write_crs(3416)


T_post = prep_da(T).rename("T").drop(["variable", "spatial_ref"])
RR_post = prep_da(RR).rename("RR").drop(["variable", "spatial_ref"])
SWE_post = prep_da(SWE).rename("SWE").drop(["variable", "spatial_ref"])
HS_post = prep_da(HS).rename("HS").drop(["variable", "spatial_ref"])
SR_post = prep_da(SR).rename("SR").drop(["variable", "spatial_ref"])
ET0_post = prep_da(ET0).rename("ET0").drop(["variable", "spatial_ref"])
dem_post = prep_da(dem).drop("spatial_ref")

# save data
T_post.rio.to_raster(out_path_T_da)
RR_post.rio.to_raster(out_path_RR_da)
SWE_post.rio.to_raster(out_path_SWE_da)
HS_post.rio.to_raster(out_path_HS_da)
SR_post.rio.to_raster(out_path_SR_da)
ET0_post.rio.to_raster(out_path_ET0_da)
dem_post.rio.to_raster(out_path_dem_da)

T_df = T_post.to_dataframe().reset_index()
RR_df = RR_post.to_dataframe().reset_index()
SWE_df = SWE_post.to_dataframe().reset_index()
HS_df = HS_post.to_dataframe().reset_index()
SR_df = SR_post.to_dataframe().reset_index()
ET0_df = ET0_post.to_dataframe().reset_index()
dem_df = dem_post.to_dataframe().reset_index()

# save with feather
T_df.to_feather(out_path_T_df)
RR_df.to_feather(out_path_RR_df)
SWE_df.to_feather(out_path_SWE_df)
HS_df.to_feather(out_path_HS_df)
SR_df.to_feather(out_path_SR_df)
ET0_df.to_feather(out_path_ET0_df)
dem_df.to_feather(out_path_dem_df)
