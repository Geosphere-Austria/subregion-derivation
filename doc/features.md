# Features

## Climate Indices

Climate indices are computed on the basis of meteorological progenitor parameters.
These progenitor parameters are available from 1961 onwards on a daily basis, and on
a consistent grid across Austria with a spatial resolution of 1 km.

The original temporal resolution of the climate indices used depends on the index
definition. Many indices are already defined for a time period of one year.
Most other indicators are aggregated to annual resolution.
Indicators with monthly resolution are treated as a separate variable for each month
(e.g.: `ETR` &rarr; `ETR_01`, `ETR_02`, ..., `ETR_12`).
Indicators defined on a daily basis are aggregated annually using both mean and standard
deviation as aggregation functions.
For example, the 30-day standardized precipitation-evaporation index is computed according
to the index definition, and is then aggregated to an annual resolution as `SPEI30_yearmean`
and `SPEI30_yearstd`, respectively.

| variable                                                                          | abbreviation                   | temporal resolution | source    | progenitor parameter        |
| :-------------------------------------------------------------------------------- | :----------------------------- | :-----------------: | :-------: | :-------------------------- |
| daily temperature range                                                           | DTR                            | y                   | SPARTACUS | temperature                 |
| growing season length                                                             | GSL                            | y                   | SPARTACUS | temperature                 |
| frost days                                                                        | FD                             | y                   | SPARTACUS | temperature                 |
| heat days                                                                         | HD                             | y                   | SPARTACUS | temperature                 |
| icing days                                                                        | ID                             | y                   | SPARTACUS | temperature                 |
| summer days                                                                       | SU                             | y                   | SPARTACUS | temperature                 |
| tropical nights                                                                   | TR                             | y                   | SPARTACUS | temperature                 |
| annual mean of daily temperature                                                  | T_yearmean                     | y                   | SPARTACUS | temperature                 |
| annual mean of minimum daily temperature                                          | TN_yearmean                    | y                   | SPARTACUS | temperature                 |
| extreme temperature range in month {01..12}                                       | ETR_{01..12}                   | m                   | SPARTACUS | temperature                 |
| minimum daily minmum temperature in month {01..12}                                | TNn_{01..12}                   | m                   | SPARTACUS | temperature                 |
| maximum daily minimum temperature in month {01..12}                               | TNx_{01..12}                   | m                   | SPARTACUS | temperature                 |
| minimum daily maximum temperature in month {01..12}                               | TXn_{01..12}                   | m                   | SPARTACUS | temperature                 |
| maximum daily maximum temperature in month {01..12}                               | TXx_{01..12}                   | m                   | SPARTACUS | temperature                 |
| consecutive dry days                                                              | CDD                            | y                   | SPARTACUS | precipitation               |
| consecutive wet days                                                              | CWD                            | y                   | SPARTACUS | precipitation               |
| annual total precipitation                                                        | PRCPTOT                        | y                   | SPARTACUS | precipitation               |
| annual count of days with precipitation > 10mm                                    | R10mm                          | y                   | SPARTACUS | precipitation               |
| annual count of days with precipitation > 20mm                                    | R20mm                          | y                   | SPARTACUS | precipitation               |
| annual mean of precipitation                                                      | RR_yearmean                    | y                   | SPARTACUS | precipitation               |
| maximum one day precipitation                                                     | Rx1day                         | y                   | SPARTACUS | precipitation               |
| maximum five day precipitation                                                    | Rx5day                         | y                   | SPARTACUS | precipitation               |
| precipitation concentration index                                                 | pci                            | y                   | SPARTACUS | precipitation               |
| antecedent precipitation index (p = 0.935; k = 30)                                | api_p0.935_k30_yearmean        | y                   | SPARTACUS | precipitation               |
| antecedent precipitation index (p = 0.935; k = 30)                                | api_p0.935_k30_yearstd         | y                   | SPARTACUS | precipitation               |
| antecedent precipitation index (p = 0.935; k = 7)                                 | api_p0.935_k7_yearmean         | y                   | SPARTACUS | precipitation               |
| antecedent precipitation index (p = 0.935; k = 7)                                 | api_p0.935_k7_yearstd          | y                   | SPARTACUS | precipitation               |
| simple precipitation intensity index                                              | SDII                           | y                   | SPARTACUS | precipitation               |
| annual mean of absolute sunshine duration                                         | SA_yearmean                    | y                   | SPARTACUS | sunshine duration           |
| annual standard deviation of absolute sunshine duration                           | SA_yearstd                     | y                   | SPARTACUS | sunshine duration           |
| annual mean of relative sunshine duration                                         | SR_yearmean                    | y                   | SPARTACUS | sunshine duration           |
| annual standard deviation of relative sunshine duration                           | SR_yearstd                     | y                   | SPARTACUS | sunshine duration           |
| annual count of days with snow depth daily difference > 0.02 m                    | snow_depth_diff_count_gt_0.02  | y                   | SNOWGRID  | snow height                 |
| annual count of days with snow depth daily difference < -0.03 m                   | snow_depth_diff_count_lt_-0.03 | y                   | SNOWGRID  | snow height                 |
| annual mean of snow depth daily difference                                        | snow_depth_diff_yearmean       | y                   | SNOWGRID  | snow height                 |
| annual standard deviation of snow depth daily difference                          | snow_depth_diff_yearstd        | y                   | SNOWGRID  | snow height                 |
| annual count of days with snow depth > 1mm                                        | snow_depth_gt_1                | y                   | SNOWGRID  | snow height                 |
| annual mean of surface snow thickness                                             | snow_depth_yearmean            | y                   | SNOWGRID  | snow height                 |
| annual standard deviation of surface snow thickness                               | snow_depth_yearstd             | y                   | SNOWGRID  | snow height                 |
| annual count of days with snow water equivalent > 3 kg m<sup>-2</sup>             | swe_tot_diff_count_gt_3        | y                   | SNOWGRID  | snow water equivalent       |
| annual count of days with snow water equivalent < -5 kg m<sup>-2</sup>            | swe_tot_diff_count_lt_-5       | y                   | SNOWGRID  | snow water equivalent       |
| annual mean of snow water equivalent daily difference                             | swe_tot_diff_yearmean          | y                   | SNOWGRID  | snow water equivalent       |
| annual standard deviation of snow water equivalent daily difference               | swe_tot_diff_yearstd           | y                   | SNOWGRID  | snow water equivalent       |
| annual mean of snow water equivalent                                              | swe_tot_yearmean               | y                   | SNOWGRID  | snow water equivalent       |
| annual standard deviation of snow water equivalent                                | swe_tot_yearstd                | y                   | SNOWGRID  | snow water equivalent       |
| annual mean temperature                                                           | bio1                           | y                   | SPARTACUS | temperature                 |
| annual mean of monthly mean diurnal range                                         | bio2                           | y                   | SPARTACUS | temperature                 |
| isothermality                                                                     | bio3                           | y                   | SPARTACUS | temperature                 |
| temperature seasonality                                                           | bio4                           | y                   | SPARTACUS | temperature                 |
| maximum temperature of warmest month                                              | bio5                           | y                   | SPARTACUS | temperature                 |
| minimum temperature of coldest month                                              | bio6                           | y                   | SPARTACUS | temperature                 |
| temperature annual range                                                          | bio7                           | y                   | SPARTACUS | temperature                 |
| mean temperature of wettest quarter                                               | bio8                           | y                   | SPARTACUS | temperature & precipitation |
| mean temperature of driest quarter                                                | bio9                           | y                   | SPARTACUS | temperature & precipitation |
| mean temperature of warmest quarter                                               | bio10                          | y                   | SPARTACUS | temperature & precipitation |
| mean temperature of coldest quarter                                               | bio11                          | y                   | SPARTACUS | temperature & precipitation |
| annual precipitation                                                              | bio12                          | y                   | SPARTACUS | precipitation               |
| precipitation of wettest month                                                    | bio13                          | y                   | SPARTACUS | precipitation               |
| precipitation of driest month                                                     | bio14                          | y                   | SPARTACUS | precipitation               |
| precipitation seasonality                                                         | bio15                          | y                   | SPARTACUS | precipitation               |
| precipitation of wettest quarter                                                  | bio16                          | y                   | SPARTACUS | precipitation               |
| precipitation of driest quarter                                                   | bio17                          | y                   | SPARTACUS | precipitation               |
| precipitation of warmest quarter                                                  | bio18                          | y                   | SPARTACUS | precipitation               |
| precipitation of coldest quarter                                                  | bio19                          | y                   | SPARTACUS | precipitation               |
| annual mean of reference evapotranspiration                                       | ET0_yearmean                   | y                   | WINFORE   | evapotranspiration          |
| annual standard deviation of reference evapotranspiration                         | ET0_yearstd                    | y                   | WINFORE   | evapotranspiration          |
| annual mean of 30-day standardized precipitation-evaporation index                | SPEI30_yearmean                | y                   | WINFORE   | SPEI                        |
| annual standard deviation of 30-day standardized precipitation-evaporation index  | SPEI30_yearstd                 | y                   | WINFORE   | SPEI                        |
| annual mean of 90-day standardized precipitation-evaporation index                | SPEI90_yearmean                | y                   | WINFORE   | SPEI                        |
| annual standard deviation of 90-day standardized precipitation-evaporation index  | SPEI90_yearstd                 | y                   | WINFORE   | SPEI                        |
| annual mean of 365-day standardized precipitation-evaporation index               | SPEI365_yearmean               | y                   | WINFORE   | SPEI                        |
| annual standard deviation of 365-day standardized precipitation-evaporation index | SPEI365_yearstd                | y                   | WINFORE   | SPEI                        |


## Geomorphometric Indices

Geomorphometric indices are derived from an ALS-DTM with a spatial resolution of 10 meters using
[`gdaldem`](https://gdal.org/programs/gdaldem.html) and [SAGA GIS](https://saga-gis.sourceforge.io/).

The data set has been reprojected from
MGI / Austria Lambert ([EPSG:31287](https://epsg.io/31287)) to
ETRS89 / Austria Lambert ([EPSG:3416](https://epsg.io/3416)) to
match the CRS of the climate data sets.

The computed terrain indices have been aggregated using different aggregation functions as available in gdal ≥ 3.3.0.

See https://gitlab.com/Rexthor/dtm-processing for details.

| Feature name                  | Filename appendix    | Tool       | Documentaton                                                               |
| :---------------------------: | :------------------: | :--------: | :------------------------------------------------------------------------: |
| Aspect                        | `aspect`             | `gdaldem`  |                                                                            |
| Catchment Area                | `catchment-area`     |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_23.html   |
| Channel Network               | `channel-network`    |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_channels_0.html     |
| Convergence Index             | `convergence-index`  |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_1.html  |
| Terrain Surface Convexity     | `convexity`          |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_21.html |
| Curvature (max)               | `curv-max`           |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_23.html |
| Curvature (min)               | `curv-min`           |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_23.html |
| Curvature (plan)              | `curv-plan`          |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_23.html |
| Curvature (profile)           | `curv-prof`          |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_23.html |
| Diurnal Anisotropic Heat      | `DAH`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_12.html |
| Flow Accumulation             | `flow-accumulation`  |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_0.html    |
| Flow Path Length              | `flow-path-length`   |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_6.html    |
| Flow Width                    | `flow-width`         |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_19.html   |
| Geomorphons                   | `geomorphons`        |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_lighting_8.html     |
| Maximum Height                | `maximum-height`     |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_23.html   |
| Modified Catchment Area       | `mod-catchment-area` |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_15.html   |
| Melton Roughness Number       | `MRN`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_23.html   |
| Negative Topographic Openness | `NTO`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_lighting_5.html     |
| Positive Topographic Openness | `PTO`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_lighting_5.html     |
| Roughness                     | `roughness`          | `gdaldem`  |                                                                            |
| Specific Catchment Area       | `SCA`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_19.html   |
| Slope (radian)                | `slope-rad`          |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_0.html  |
| Slope (degree)                | `slope`              | `gdaldem`  |                                                                            |
| Stream Power Index            | `SPI`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_21.html   |
| Sky View Factor               | `SVF`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_lighting_3.html     |
| Topographic Position Index    | `TRI`                | `gdaldem`  |                                                                            |
| Topographic Roughness Index   | `TRI`                | `gdaldem`  |                                                                            |
| Topographic Wetness Index     | `TWI`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_hydrology_15.html   |
| Vector Ruggedness Measure     | `VRM`                |  SAGA GIS  | https://saga-gis.sourceforge.io/saga_tool_doc/8.0.0/ta_morphometry_17.html | 
