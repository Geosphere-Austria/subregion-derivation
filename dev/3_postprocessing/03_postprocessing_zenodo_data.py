"""script to assign attributes to the core variable output data for publishing on zenodo"""

import xarray as xr
from ..utils import ordered_variable_list

fname_in = "dat/interim/02_preprocessed_climate_normals/indicators_climate_normals.nc"
fname_out = "dat/out/physioclimatic_features_grid_AT_average_1992-2021.nc"

# select input features
da_in = (
    xr.load_dataarray(fname_in)
    .sel(metric="mean")
    .isel(climate_period=1)
    .sel(variable=ordered_variable_list)
)

# save some existing information
metric = da_in.metric.values
climate_period = da_in.climate_period.values

# drop all nonconformal elements
da_in = da_in.drop(["metric", "climate_period"])

da_out = da_in.copy()

# set attrs for x coordinate
da_out.x.attrs["units"] = "m"
da_out.x.attrs["long_name"] = "x coordinate of projection"
da_out.x.attrs["standard_name"] = "projection_x_coordinate"
da_out.x.attrs["axis"] = "X"
da_out.x.attrs["projection"] = "ETRS89 / Austria Lambert - EPSG:3416"

# set attrs for y coordinate
da_out.y.attrs["units"] = "m"
da_out.y.attrs["long_name"] = "y coordinate of projection"
da_out.y.attrs["standard_name"] = "projection_y_coordinate"
da_out.y.attrs["axis"] = "Y"
da_out.y.attrs["projection"] = "ETRS89 / Austria Lambert - EPSG:3416"

# set attrs for variable coordinate
da_out.y.attrs["long_name"] = "str of variable names"

# set global attrs
da_out.attrs["time_aggregation_method"] = metric
da_out.attrs["climate_normal"] = climate_period
da_out.attrs["source"] = (
    "calculation and aggregation of different physioclimatic indices"
)
# da_out.attrs["reference"] = ""
da_out.attrs["contact"] = "Sebastian Lehner (sebastian.lehner@geosphere.at)"
da_out.attrs["creation_date"] = "2023-05-26"
da_out.attrs["projection"] = "ETRS89 / Austria Lambert - EPSG:3416"

da_out.name = "features"

da_out.to_netcdf(fname_out)
