"""Calculate climate normals for the use in clustering"""
from pathlib import Path

import numpy as np
from scipy.ndimage.filters import generic_filter as gen_filter
import xarray as xr


def filter_nanmean(da_: xr.DataArray, size: int = 20):
    """2d filter to be applied on the input xr.DataArray with
    a 2d box side length defined by size."""
    da_in = da_.copy()
    da_list = []
    for var_iter in da_in["variable"]:
        da_iter = da_in.sel(variable=var_iter)
        da_iter.data = gen_filter(da_iter, np.nanmean, size=size, mode="nearest")
        da_list.append(da_iter)
    da_out = xr.concat(da_list, dim="variable")
    return da_out.where(~np.isnan(da_), np.nan)


if __name__ == "__main__":

    filter_ = True

    data_dir = Path("dat/raw/grid/obs")
    tmp_dir = Path("dat/interim")

    clim_period_1 = slice("01-01-1962", "31-12-1991")
    clim_period_2 = slice("01-01-1992", "31-12-2021")
    clim_1_str = "01-01-1962 to 31-12-1991"
    clim_2_str = "01-01-1992 to 31-12-2021"

    data_list = []
    for file in Path(tmp_dir, "01_concat_files").rglob("*concat*.nc"):
        print(f"..Processing {file.name = }..")
        with xr.load_dataarray(file) as da_load:
            if file.name == "dtm_concat_indices.nc":
                # dtm data is time invariant, hence is just the input data
                clim_mean_1 = da_load.copy()
                clim_mean_2 = da_load.copy()
                clim_std_1 = da_load.copy()
                clim_std_2 = da_load.copy()
            else:
                # else for all other data for which climate normals can be calculated
                clim_mean_1 = da_load.sel(time=clim_period_1).mean(dim="time")
                clim_mean_2 = da_load.sel(time=clim_period_2).mean(dim="time")
                clim_std_1 = da_load.sel(time=clim_period_1).std(dim="time")
                clim_std_2 = da_load.sel(time=clim_period_2).std(dim="time")

            clim_mean_1 = clim_mean_1.assign_coords(
                {"climate_period": clim_1_str, "metric": "mean"}
            )
            clim_mean_2 = clim_mean_2.assign_coords(
                {"climate_period": clim_2_str, "metric": "mean"}
            )
            clim_std_1 = clim_std_1.assign_coords(
                {"climate_period": clim_1_str, "metric": "std"}
            )
            clim_std_2 = clim_std_2.assign_coords(
                {"climate_period": clim_2_str, "metric": "std"}
            )

            if not file.name == "misc_concat_snowgrid_indices.nc":
                # grab grid for spartacus v2.1 that is needed for later snowgrid
                # interpolation
                ds_grid_out = xr.Dataset(
                    {
                        "x": (["x"], da_load.coords["x"].values),
                        "y": (["y"], da_load.coords["y"].values),
                    }
                )
            else:
                # inteprolate snowgrid from sparta v1.5 grid to v2.1
                ds_grid_in = xr.Dataset(
                    {
                        "x": (["x"], da_load.coords["x"].values),
                        "y": (["y"], da_load.coords["y"].values),
                    }
                )
                # pylance complains, but ds_grid_out will always exist
                # because of the way the files are globbed
                clim_mean_1 = clim_mean_1.interp_like(ds_grid_out)
                clim_mean_2 = clim_mean_2.interp_like(ds_grid_out)
                clim_std_1 = clim_std_1.interp_like(ds_grid_out)
                clim_std_2 = clim_std_2.interp_like(ds_grid_out)

            if not file.name == "dtm_concat_indices.nc":
                # smooth field because of technical vs. physical resolution
                if filter_:
                    clim_mean_1 = filter_nanmean(clim_mean_1)
                    clim_mean_2 = filter_nanmean(clim_mean_2)
                    clim_std_1 = filter_nanmean(clim_std_1)
                    clim_std_2 = filter_nanmean(clim_std_2)

            data_iter_mean = xr.concat([clim_mean_1, clim_mean_2], dim="climate_period")
            data_iter_std = xr.concat([clim_std_1, clim_std_2], dim="climate_period")
            data_iter_fin = xr.concat([data_iter_mean, data_iter_std], dim="metric")
            data_list.append(data_iter_fin)

    print("..Concatenating data..")
    da_merged = xr.concat(data_list, dim="variable")
    new_par_dir = Path(tmp_dir, "02_preprocessed_climate_normals")
    new_par_dir.mkdir(exist_ok=True, parents=True)
    print("..Saving data..")
    if filter_:
        da_merged.to_netcdf(Path(new_par_dir, "indicators_climate_normals.nc"))
    else:
        da_merged.to_netcdf(
            Path(new_par_dir, "indicators_climate_normals_unfiltered.nc")
        )
