"""script to generate heatmap plots"""
from pathlib import Path
from utils import (
    load_climate_normals_and_reshape,
    drop_vars_with_too_many_nans,
    ordered_variable_list,
)
import proplot as pplt
import numpy as np
import xarray as xr


def get_clean_variables_files(STR_SUFFIX: str):
    """short function to load and process climate indicator variables"""
    variables_file = Path(
        f"dat/interim/02_preprocessed_climate_normals/indicators_climate_normals{STR_SUFFIX}.nc"
    )
    xrda = load_climate_normals_and_reshape(
        file_path=variables_file, metric="mean", climate_period=1
    )
    xrda = drop_vars_with_too_many_nans(da_in=xrda)
    return xrda.dropna(dim="z")


def calc_pearson(xrda):
    return np.corrcoef(xrda.T.values)


heatmap_kwargs = {
    "cmap": "Spectral_r",
    "colorbar": "r",
    "vmin": -1,
    "vmax": 1,
}


def plot_categories_heatmap(corrs_, title_method, out_file, version):
    idx = np.cumsum([80, 20, 4, 13, 10])
    corrs_ = np.insert(corrs_, idx, np.nan, axis=0)
    corrs_ = np.insert(corrs_, idx, np.nan, axis=1)

    fig, ax = pplt.subplots(refwidth=4.5)
    ax.heatmap(corrs_, **heatmap_kwargs)

    ax.format(
        suptitle=f"{title_method} correlation coefficients for mean climate indicators"
    )
    cat_labels = [
        "temperature",
        "precipitation",
        "radiation",
        "snow",
        "combined",
        "geomorphometry",
        "",
    ]
    if version == "v2":
        cat_len = [-0.5, 80, 20, 4, 13, 10, 49]
        cat_len = np.cumsum(cat_len) + range(7)
        ax.format(ylocator=cat_len, xlocator=[], yformatter=cat_labels)
    elif version == "v3":
        cat_len = [-0.5, 79.5, 19.75, 4.25, 12.75, 10.25, 48.5]
        cat_len = np.cumsum(cat_len) + range(7)
        ax.format(ylocator=cat_len, xlocator=cat_len, xformatter=[], yformatter=[])
        locs = [40, 90.5, 103.5, 113, 125.5, 153]
        for loc, cat in zip(locs, cat_labels):
            ax.text(-3, loc, cat, ha="right", va="center")
            ax.text(loc + 3, -1, cat, ha="right", va="top", rotation=45)
    fig.save(out_file)


def plot_core_vars(corrs_, labels, title_method, out_file, corr_th: float):
    if title_method == "Pearson":
        fig, ax = pplt.subplots(refwidth=5)
        ax.heatmap(corrs_, **heatmap_kwargs)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        latex_str = r"$\rho_\theta=$"
        ax.format(
            suptitle=f"{title_method} correlation coefficient for core variables, {latex_str}{corr_th}"
        )
        fig.save(out_file)
    else:
        fig, axs = pplt.subplots(refwidth=5, nrows=2, sharex=False, sharey=False)
        axs_iter = iter(axs)

        for i in range(2):
            ax = next(axs_iter)
            step_len = 40
            step = i * step_len
            ax.heatmap(
                corrs_[step : (step + step_len), step : (step + step_len)],
                **heatmap_kwargs,
            )
            ax.set_yticklabels(labels[step : (step + step_len)])

        ax.format(
            suptitle=f"{title_method} correlation coefficients for mean climate indicators in 30 tick steps along diagonal"
        )
        fig.save(out_file)


if __name__ == "__main__":
    plot_all_vars_bool = False
    plot_core_vars_bool = True

    for filter_, str_suffix in zip([True, False], ["", "_unfiltered"]):
        print(f"..processing with filter == {filter_}..")
        if plot_all_vars_bool or plot_core_vars_bool:
            da_clean = get_clean_variables_files(STR_SUFFIX=str_suffix)
            da_clean = da_clean.sel(variable=ordered_variable_list)

            pearson_correlations = calc_pearson(xrda=da_clean)

        if plot_all_vars_bool:
            plot_categories_heatmap(
                corrs_=pearson_correlations,
                title_method="Pearson",
                out_file=f"plt/heatmap/heatmap_all_vars_v3{str_suffix}.png",
                version="v3",
            )

        for correlation_threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            print(f"..processing {correlation_threshold = }..")
            if plot_core_vars_bool:
                da_core = xr.load_dataarray(
                    f"dat/interim/03_feature_definition{str_suffix}/"
                    f"core_variables_mean_t1_r{correlation_threshold}.nc"
                )
                da_core = da_core.stack(z=["x", "y"]).T.dropna(dim="z")
                pearson_labels = sorted(
                    da_core.coords["variable"].values, key=str.casefold
                )
                da_core_ordered = da_core.sel(variable=pearson_labels)
                pearson_correlations_core = calc_pearson(xrda=da_core_ordered)
                plot_core_vars(
                    corrs_=pearson_correlations_core,
                    labels=pearson_labels,
                    title_method="Pearson",
                    out_file=f"plt/heatmap/heatmap_core_r{correlation_threshold}_v3{str_suffix}.png",
                    corr_th=correlation_threshold,
                )
