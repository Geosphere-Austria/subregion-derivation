"""preprocess data for the data-driven subregion derivation
processed data is saved under /dat/interim
"""
from pathlib import Path

import pandas as pd
import xarray as xr


def load_concat_climdex_data(parent_dir: Path) -> xr.DataArray:
    """Loads all climdex data sets and concatenates them into one xr.DataArray"""
    climdex_dir = Path(parent_dir, "climate_indices", "climdex")
    data_list = []
    for file in climdex_dir.rglob("*.nc"):
        da_iter = xr.open_dataarray(file)
        print(f"..grab variable {da_iter.name = }..")
        new_time = pd.date_range(
            f"1-1-{da_iter.time.dt.year.values[0]}",
            f"1-1-{da_iter.time.dt.year.values[-1]+1}",
            freq="Y",
        )
        if da_iter.time.shape[0] > 100:  # triggers for monthly data
            for month, da_group in da_iter.groupby("time.month"):
                variable_name = f"{da_iter.name}_{str(month).zfill(2)}"
                da_group["time"] = new_time
                da_group.name = "climate_indicator"
                da_group = da_group.assign_coords({"variable": variable_name})
                data_list.append(da_group)
        else:
            variable_name = da_iter.name
            da_iter["time"] = new_time
            da_iter.name = "climate_indicator"
            da_iter = da_iter.assign_coords({"variable": variable_name})
            data_list.append(da_iter)

    return xr.concat(data_list, dim="variable")


def load_concat_misc_data(parent_dir: Path, key: str) -> xr.DataArray:
    """load and concatenate data from misc climate indicators"""
    if key == "spartacus_winfore":
        key_list = [
            "api_aggregates",
            "pci",
            "ET0",
            "SPEI30",
            "SPEI90",
            "SPEI365",
            "SA",
            "SR",
        ]
    elif key == "snowgrid":
        key_list = [
            "snow_depth",
            "snow_depth_diff",
            "swe_tot",
            "swe_tot_diff",
        ]

    data_list = []
    for variable in key_list:
        misc_dir = Path(parent_dir, "climate_indices", "misc", variable)
        for file in misc_dir.rglob("*.nc"):
            with xr.load_dataarray(file) as da_iter:
                if da_iter.name == "api":
                    p_decay = file.name.split("_")[1]
                    k_timeperiod = file.name.split("_")[2]
                    metric = file.name.split("_")[3]
                    variable_name = f"{da_iter.name}_{p_decay}_{k_timeperiod}_{metric}"
                else:
                    variable_name = da_iter.name

                # if snowgrid drop lat/lon
                if key == "snowgrid":
                    da_iter = da_iter.drop_vars(["lat", "lon"])
                # if spartacus_winfore drop quantile
                if key == "spartacus_winfore":
                    try:
                        da_iter = da_iter.drop_vars("quantile")
                    except ValueError:
                        pass

                print(f"..grab variable {variable_name = }..")
                new_time = pd.date_range(
                    f"1-1-{da_iter.time.dt.year.values[0]}",
                    f"1-1-{da_iter.time.dt.year.values[-1]+1}",
                    freq="Y",
                )
                da_iter["time"] = new_time
                da_iter.name = "climate_indicator"
                da_iter = da_iter.assign_coords({"variable": variable_name})
                data_list.append(da_iter.load())
    return xr.concat(data_list, dim="variable")


def load_concat_bioclim_data(parent_dir: Path) -> xr.DataArray:
    """load and process bioclim data"""
    bioclim_dir = Path(parent_dir, "climate_indices", "bioclim")
    data_list = []
    for file in bioclim_dir.rglob("bio*.nc"):
        with xr.load_dataset(file, decode_times=False) as ds_iter:
            variable_name = file.name.split("_")[0]
            da_iter = ds_iter[variable_name]
            print(f"..grab variable {variable_name = }..")
            # year hardcoded, because of source data
            time_len = da_iter.time.shape[0]
            time_start = 1961
            new_time = pd.date_range(
                f"1-1-{time_start}",
                f"1-1-{time_start + time_len}",
                freq="Y",
            )
            da_iter["time"] = new_time
            da_iter.name = "climate_indicator"
            da_iter = da_iter.assign_coords({"variable": variable_name})
            data_list.append(da_iter.load())
    return xr.concat(data_list, dim="variable")


def load_concat_dtm_data(parent_dir: Path) -> xr.DataArray:
    """load, preprocess and concatenate dtm data"""
    data_list = []
    for file in parent_dir.rglob("dtm_austria*.tif"):
        variable_name = (
            f"dtm_{file.name.replace('dtm_austria_', '').replace('.tif', '')}"
        )
        print(f"..grab variable {variable_name = }..")
        with xr.load_dataarray(file) as da_iter:
            da_iter = da_iter.squeeze().drop_vars(["band", "spatial_ref"])
            da_iter.name = "climate_indicator"
            da_iter = da_iter.assign_coords({"variable": variable_name})
            data_list.append(da_iter.load())
    return xr.concat(data_list, dim="variable")


if __name__ == "__main__":
    data_dir = Path("dat/raw/grid/obs")
    data_dir_dtm = Path("dat/raw/grid/dtm/dtm_aggregates/dtm_spartacus")
    tmp_dir = Path("dat/interim")

    # preprocess climdex indicators
    climdex_file = Path(tmp_dir, "01_concat_files", "climdex_concat_indices.nc")
    if climdex_file.exists():
        print(f".. {climdex_file = } already exists, skipping..")
    else:
        da_climdex = load_concat_climdex_data(parent_dir=data_dir)
        print("..saving data..")
        da_climdex.to_netcdf(climdex_file)

    # preprocess misc indicators
    for dataset in [
        "spartacus_winfore",  # sparta v2.1 grid
        "snowgrid",  # sparta v1.5 grid
    ]:  # different processing is needed because snowgrid is on sparta v1.5 grid
        misc_file = Path(
            tmp_dir, "01_concat_files", f"misc_concat_{dataset}_indices.nc"
        )
        if misc_file.exists():
            print(f".. {misc_file = } already exists, skipping..")
        else:
            print(f"..work on {misc_file.name = }..")
            da_misc = load_concat_misc_data(parent_dir=data_dir, key=dataset)
            print("..saving data..")
            da_misc.to_netcdf(misc_file)

    # preprocess bioclim indicators
    bioclim_file = Path(tmp_dir, "01_concat_files", "bioclim_concat_indices.nc")
    if bioclim_file.exists():
        print(f".. {bioclim_file = } already exists, skipping..")
    else:
        print(f"..work on {bioclim_file.name = }..")
        da_misc = load_concat_bioclim_data(parent_dir=data_dir)
        print("..saving data..")
        da_misc.to_netcdf(bioclim_file)

    # preprocess DTM data
    dtm_file = Path(tmp_dir, "01_concat_files", "dtm_concat_indices.nc")
    if dtm_file.exists():
        print(f".. {dtm_file = } already exists, skipping..")
    else:
        print(f"..work on {dtm_file.name = }..")
        da_misc = load_concat_dtm_data(parent_dir=data_dir_dtm)
        print("..saving data..")
        da_misc.to_netcdf(dtm_file)
