from pathlib import Path

import xarray as xr

data_dir = Path("./dat/raw/grid/obs")


def load_climdex_data(parent_dir: Path) -> xr.DataArray:
    """Loads all climdex data sets and concatenates them into one xr.DataArray"""
    climdex_dir = Path(data_dir, "climate_indices", "climdex")
    data_list = []
    for file in climdex_dir.rglob("*.nc"):
        da = xr.load_dataarray(file)
        if da.time.shape[0] > 100:  # triggers for monthly data
            for month, da_group in da.groupby("time.month"):
                da_group.name = f"{da.name}_{str(month).zfill(2)}"
                data_list.append(da_group)
        else:
            data_list.append(da)

    da_concat = xr.concat(data_list, dim="variable")
    return da_concat


da = load_climdex_data(parent_dir=data_dir)
da.to_netcdf(Path("dat/interim", "climdex_concat.nc"))
