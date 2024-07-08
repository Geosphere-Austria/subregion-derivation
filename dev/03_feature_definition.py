"""investigate feature definition and colinearities"""
from pathlib import Path

import numpy as np

from utils import (
    load_climate_normals_and_reshape,
    drop_vars_with_too_many_nans,
    drop_vars_with_high_colinearities,
    ordered_variable_list,
)


if __name__ == "__main__":

    filter_ = True

    CLIM_PERIOD = 1
    METRIC = "mean"
    CORR_METHOD = "pearson"

    if filter_:
        str_suffix = ""
    else:
        str_suffix = "_unfiltered"

    if CORR_METHOD == "pearson":
        OUT_PATH = f"dat/interim/03_feature_definition{str_suffix}"
    else:
        OUT_PATH = f"dat/interim/03_feature_definition_{CORR_METHOD}{str_suffix}"

    variables_file = Path(
        "dat/interim/02_preprocessed_climate_normals/"
        f"indicators_climate_normals{str_suffix}.nc"
    )

    for CORRCOEF in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        da_data = load_climate_normals_and_reshape(
            file_path=variables_file, metric=METRIC, climate_period=CLIM_PERIOD
        )
        da_clean = drop_vars_with_too_many_nans(da_in=da_data)
        da_clean = da_clean.dropna(dim="z")

        # reorder input params based on deterministic order
        da_clean = da_clean.sel(variable=ordered_variable_list)

        if CORR_METHOD == "pearson":
            # pearson correlation
            correlations = np.corrcoef(da_clean.T.values)
        else:
            # kendalltau correlation
            from scipy.stats import kendalltau

            kt_list = []
            for variable in da_clean.T.variable:
                kt_list.append(
                    [
                        kendalltau(variable.values, x.values)[0]
                        for x in da_clean.T.variable
                    ]
                )
            correlations = np.array(kt_list)

        np.save(
            f"{OUT_PATH}/correlations_{METRIC}_t{CLIM_PERIOD}_{CORR_METHOD}.npy",
            correlations,
        )

        # plot correlations as histogram
        import proplot as pplt

        fig, ax = pplt.subplots()
        ax.hist(
            np.abs(correlations.ravel()),
            bins=np.arange(0, 1.01, 0.05),
            filled=True,
            alpha=0.7,
            edgecolor="gray",
        )
        ax.axvline(0.95, color="k", linestyle="--", linewidth=0.7)
        ax.format(title=CORR_METHOD)
        fig.save(f"{OUT_PATH}/hist_{CORR_METHOD}.png")

        da_core = drop_vars_with_high_colinearities(
            da_in=da_clean,
            corr=correlations,
            corr_threshold=CORRCOEF,
            json_file_path=OUT_PATH,
            metric=METRIC,
            clim_period=CLIM_PERIOD,
        )

        fname_out = f"{METRIC}_t{CLIM_PERIOD}_r{CORRCOEF}"
        da_fin = da_core.unstack("z").sortby(["y", "x"])

        len_core_vars = da_fin.coords["variable"].values.shape[0]
        print(f"threshold: {CORRCOEF = } // found {len_core_vars} core variables")

        da_fin.to_netcdf(f"{OUT_PATH}/core_variables_{fname_out}.nc")
