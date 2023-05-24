"""Contains various utility functions for subregion-derivation"""
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
import pandas as pd
import proplot as pplt
from sklearn import manifold, decomposition
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import xarray as xr
import seaborn as sns
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_climate_normals_and_reshape(
    file_path: Path, metric: str = "mean", climate_period: int = 0
) -> xr.DataArray:
    """load climate normals and reshape to model-suitable shape"""
    da = xr.load_dataarray(file_path)
    da_reshaped = (
        da.stack(z=["x", "y"]).sel(metric=metric).isel(climate_period=climate_period).T
    )
    return da_reshaped


def load_core_variables_and_reshape(file_path: Path) -> xr.DataArray:
    """load core variables defined by the use of corrcoeff;
    see 03_feature_definition.py
    """
    return xr.load_dataarray(file_path).stack(z=["x", "y"]).T


def create_empty_dataset_for_clusters(
    da_original: xr.DataArray, methods_list: List[str]
) -> xr.Dataset:
    """Convenience function to assign results of clustering to xr.Dataset
    Assumes the data in da_original are stacked in dimension 'z'
    """
    ds_grid = xr.Dataset({"z": da_original.z})
    return ds_grid.assign(
        variables={
            method: (("z"), np.nan * np.empty(ds_grid.z.shape))
            for method in methods_list
        }
    )


def add_cluster_result_to_datasets(
    ds_in: xr.Dataset, cluster: xr.Dataset
) -> xr.Dataset:
    """Convenience function to assign values to input xr.Dataset for the specified variable
    Assumes the data are stacked in dimension 'z'
    """
    ds_out = ds_in.copy()
    cluster_method = list(cluster.keys())[0]
    ds_out[cluster_method].loc[cluster.z] = cluster[cluster_method].values
    return ds_out


def calc_kmeans_clusters(data: xr.DataArray, n_clusters: int = 5) -> xr.Dataset:
    """Calculate kmeans clusters via sci-kit learn
    Assumes the samples in data are stacked in dimension 'z'
    """
    pca_kmeans = decomposition.PCA(n_components=n_clusters).fit(data)
    kmeans = KMeans(init=pca_kmeans.components_, n_clusters=n_clusters, n_init=1)
    results = kmeans.fit_predict(data)

    ds_data = xr.Dataset({"z": data.z})
    ds_data = ds_data.assign(variables={"kmeans": (("z"), results + 1)})
    return ds_data


def calc_spectral_clustering(data: xr.DataArray, n_clusters: int = 5) -> xr.Dataset:
    """Calculate clusters via SpectralClustering using sci-kit learn.
    Assumes the samples in data are stacked in dimension 'z'
    """
    spec_clust = SpectralClustering(
        n_clusters=n_clusters, random_state=0, assign_labels="discretize"
    )
    results = spec_clust.fit_predict(data)

    ds_data = xr.Dataset({"z": data.z})
    ds_data = ds_data.assign(variables={"spectral_clustering": (("z"), results + 1)})
    return ds_data


def calc_dbscan_clusters(
    data: xr.DataArray, eps: float = 0.5, min_samples: int = 5
) -> xr.Dataset:
    """Calculate clusters via DBSCAN using sci-kit learn.
    Assumes the samples in data are stacked in dimension 'z'
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    results = dbscan.fit_predict(data)

    ds_data = xr.Dataset({"z": data.z})
    ds_data = ds_data.assign(variables={"dbscan": (("z"), results + 1)})
    return ds_data


def calc_tsne(
    data: xr.DataArray,
    dim_embedding_space: int = 2,
    perplexity: int = 100,
    n_iter: int = 500,
) -> np.array:
    """Calculate a representation in an embedding space via tsne"""
    tsne = manifold.TSNE(
        n_components=dim_embedding_space,
        init="pca",
        random_state=0,
        perplexity=perplexity,
        learning_rate="auto",
        n_iter=n_iter,
    )
    results = tsne.fit_transform(data)
    return results


def plot_cluster(data: xr.DataArray, save: bool = False) -> None:
    """plotting routine for cluster visualization"""
    unique_vals = np.unique(data)
    unique_vals = unique_vals[~np.isnan(unique_vals)]
    n_steps = len(unique_vals)
    fig, axs = pplt.subplots()
    axs.pcolormesh(
        data.unstack("z").T,
        colorbar="b",
        cmap="flatui",
        levels=pplt.arange(0.5, 0.5 + n_steps, 1),
        qualitative=True,
        colorbar_kw={
            "ticks": pplt.arange(1, n_steps),
            "ticklabels": [str(i) for i in range(1, 1 + n_steps)],
        },
    )
    axs.format(title=data.name)
    if save:
        fig.save(f"plt/clusters_{data.name}.png")
    return axs


def drop_vars_with_too_many_nans(
    da_in: xr.DataArray, nan_limit: int = 108245
) -> xr.DataArray:
    """drop variables in a xr.coord named 'variable' that contain more
    nans than the nan_limit.
    the nan_limit was manually derived from precipitation spartacus fields
    """
    print("check for too many nans in data:")
    print(f"pre: n_variable = {da_in.coords['variable'].values.shape[0]}")
    for variable in da_in["variable"].values:
        da_iter_ = da_in.sel(variable=variable)
        nans = da_iter_.where(np.isnan(da_iter_), drop=True).shape[0]
        if nans > nan_limit:
            print(f"drop {variable =} with {nans = }")
            da_in = da_in.drop_sel(variable=variable)
    print(f"post: n_variable = {da_in.coords['variable'].values.shape[0]}")
    return da_in


def drop_vars_with_high_colinearities(
    da_in: xr.DataArray,
    corr: np.array,
    corr_threshold: float = 0.95,
    json_file_path: str = "",
    metric: str = "",
    clim_period: str = "",
) -> xr.DataArray:
    """determine groups of high colinearities"""
    variables_arr = da_in.T.coords["variable"].values
    variables_set = set(variables_arr)
    groups = {}
    for idx, variable in enumerate(variables_arr):
        if variable in variables_set:
            corr_iter = corr[idx, idx:]
            group_set = set(
                variables_arr[np.where(np.abs(corr_iter) > corr_threshold)[0] + idx]
            )
            group_set.discard(variable)
            groups[variable] = sorted(list(group_set))
            _ = [variables_set.discard(group_set_i) for group_set_i in group_set]

    da_out = da_in.sel(variable=list(variables_set))

    with open(
        f"{json_file_path}/core_variable_groups_{metric}_t{clim_period}_r{corr_threshold}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(groups, file, indent=4)
    return da_out


def load_clean_data(
    corrcoef: float,
    time_period: int = 0,
    metric: str = "mean",
    path_to_file: Path = None,
):
    """Load core variables based on corrcoef threshold and time_period, where
    for the latter 0 is the earlier climate period and 1 is the later period.
    """
    if path_to_file:
        data_iter = load_core_variables_and_reshape(file_path=path_to_file)
    else:
        variables_file = lambda t, r: Path(
            f"dat/interim/feature_definition/core_variables_{metric}_t{t}_r{r}.nc"
        )
        data_iter = load_core_variables_and_reshape(
            file_path=variables_file(time_period, corrcoef)
        )
    return data_iter.dropna(dim="z")


def normalize_data(data: np.array) -> np.array:
    """Normalise input data."""
    return StandardScaler().fit_transform(data)


def apply_pca(data: np.array, n_pca: int) -> Tuple[np.array, float]:
    """Applies pca to the input data and returns the pca object
    and the explained variance.
    """
    pca = PCA(n_components=n_pca)
    dat_in = pca.fit_transform(data)
    explained_variance = sum(pca.explained_variance_)
    return dat_in, explained_variance


def apply_umap(data: np.array, params: dict):
    """Applies UMAP to the input data"""
    reducer = umap.UMAP(
        n_neighbors=params["umap_n_neighbors"],
        min_dist=params["umap_min_dist"],
        n_components=params["umap_n_components"],
    )
    return reducer.fit_transform(data)


def apply_hdbscan(data: np.array, params: dict):
    """Applies hdbscan to the input data"""
    return hdbscan.HDBSCAN(
        min_cluster_size=params["hs_min_cluster_size"],
        min_samples=params["hs_min_samples"],
        cluster_selection_epsilon=params["hs_cluster_selection_epsilon"],
    ).fit_predict(data)


def generate_cluster(
    r: float,
    t: float,
    pca_bool: bool,
    n_pca: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_n_components: int,
    hs_min_cluster_size: int,
    hs_cluster_selection_epsilon: float,
    hs_min_samples: int = None,
    normalisation_bool: bool = True,
    load_data: bool = True,
    data: xr.DataArray = None,
    metric_str_data: str = "mean",
):
    """calculate a cluster based on a UMAP > HDBSCAN workflow with optional
    pre-filtering via pca.

    Params:
        n_pca: dimension for pca dim-reduction
        umap_n_neighbors: controls how UMAP balances local vs global structure;
                smaller = finer details, but less 'bigger picture'
                try values from 2 to 100 (default 15)
        umap_min_dist: how tightly points are packed to each other;
                low values = more clumped embeddings;
                try lower values for stronger clustering 0 to 1 (default 0.1)
        umap_n_components: dimension of embedding space;
                does not need to be necessarily 2;
                more components help UMAP to find seperation
                between clusters (default 2)
        umap_metric: how distance is calculated (default 'euclidian')
        hs_min_cluster_size: smallest grouping that is considered a cluster
        hs_min_samples: minimum number of samples per cluster
        hs_cluster_selection_epsilon: threshold where clusters are split/not split
                into microclusters; depends on the distance between data points
    """
    if load_data:
        dat_raw = load_clean_data(corrcoef=r, time_period=t, metric=metric_str_data)
    else:
        dat_raw = data

    # normalisation
    if normalisation_bool:
        dat_pre = StandardScaler().fit_transform(dat_raw)
    else:
        dat_pre = dat_raw

    # pca
    if pca_bool:
        pca = PCA(n_components=n_pca)
        dat_in = pca.fit_transform(dat_pre)
        explained_variance = sum(pca.explained_variance_)
    else:
        dat_in = dat_pre
        explained_variance = None

    # umap
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=umap_n_components,
    )
    embedding = reducer.fit_transform(dat_in)

    # hdbscan
    clust = hdbscan.HDBSCAN(
        min_cluster_size=hs_min_cluster_size,
        min_samples=hs_min_samples,
        cluster_selection_epsilon=hs_cluster_selection_epsilon,
    ).fit_predict(embedding)
    return dat_raw, dat_in, embedding, clust, explained_variance


def plot_umap_cluster(embedding, clust, ax=None):
    if not ax:
        _, ax = pplt.subplots()
    large_clusters = (
        pd.Series(clust)
        .value_counts()
        .sort_index()[pd.Series(clust).value_counts().sort_index() > 1000]
    )
    if -1 in large_clusters.keys():
        large_clusters = large_clusters.drop(-1)

    len_of_large_clusters = len(large_clusters)
    palette = sns.color_palette("deep", len_of_large_clusters)
    c_dict = {key: (0.5, 0.5, 0.5) for key in np.unique(clust)}
    c_dict[-1] = (0.0, 0.0, 0.0)  # -1 is noise
    for key_large, pal in zip(large_clusters.keys(), palette):
        c_dict[key_large] = pal

    colors = [c_dict[x] for x in clust]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.1, s=2)
    ax.format(title="hdbscan clustering on umap projection in 2D")
    return ax


def plot_composit_umap_hdbscan(
    dat_raw: xr.DataArray,
    embedding: np.array,
    clust: np.array,
    params: dict,
    savepath: Path = None,
    colors_dict=None,
):
    """Plot hdbscan cluster class distribution, umap embedding and
    spartacus grid cluster results.
    """
    fig, axes = pplt.subplots(ncols=3, sharex=False, sharey=False, wratios=(1, 1, 2))
    axes.format(abc="a.", abcloc="ul")

    large_clusters = (
        pd.Series(clust)
        .value_counts()
        .sort_index()[
            pd.Series(clust).value_counts().sort_index() > params["hs_min_cluster_size"]
        ]
    )
    if -1 in large_clusters.keys():
        large_clusters = large_clusters.drop(-1)

    # remap large_clusters according to cluster size for consistent coloring
    old_index = large_clusters.sort_values(ascending=False).index
    large_clusters = (
        large_clusters.sort_values(ascending=False).reset_index().drop("index", axis=1)
    )

    remap_dict = {
        key: value for key, value in zip(old_index, range(old_index.shape[0]))
    }
    remapped_clust = np.copy(clust)
    for key, value in remap_dict.items():
        remapped_clust[clust == key] = value

    len_of_large_clusters = len(large_clusters)
    if colors_dict:
        c_dict = {}
        for key_clr, clr in colors_dict.items():
            c_dict[key_clr] = clr
    else:
        palette = sns.color_palette("deep", len_of_large_clusters)
        c_dict = {key: (0.5, 0.5, 0.5) for key in np.unique(clust)}
        c_dict[-1] = (0.0, 0.0, 0.0)  # -1 is noise
        for key_large, pal in zip(large_clusters.index, palette):
            c_dict[key_large] = pal

    colors = [c_dict[x] for x in remapped_clust]
    axes[1].scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.1, s=2)
    axes[1].format(
        title="HDBSCAN clustering on UMAP projection",
        xlabel="UMAP dimension 1 coordinate",
        ylabel="UMAP dimension 2 coordinate",
    )

    axes[0].bar(
        pd.Series(remapped_clust).value_counts().sort_index(),
        color=list(c_dict.values()),
    )
    ylim_upper = max(int(pd.Series(clust).value_counts().max()), 35_000)
    axes[0].format(
        title="Histogram of classes (-1 = noise)",
        ylabel="Number of gridpoints",
        xlabel="Cluster classes",
        xtickminor=False,
        xticks=list(range(-1, len_of_large_clusters)),
        xticklabels=[
            str(entry)
            for entry in (
                [
                    -1,
                ]
                + list(range(1, len_of_large_clusters + 1))
            )
        ],
        ylim=[0, ylim_upper],
    )

    ## transform back into x, y
    try:
        xy_clust = dat_raw.isel(variable=0).copy()
    except ValueError:
        xy_clust = dat_raw.copy()
    xy_clust["variable"] = "umap_hdbscan"
    xy_clust.values = clust
    data = xy_clust
    axes[2].scatter(
        data.x,
        data.y,
        c=colors,
        s=2,
    )
    axes[2].set_title(
        f"HDBSCAN clustering transformed back into xy-Coordinates (key = {params['key']})"
    )
    axes[2].format(aspect="equal")
    first_line = (
        f"Pipeline: core_variables (r = {params['corr']}) > PCA > UMAP > HDBSCAN"
    )
    second_line_vars = [
        r"$dim_{pca}=$",
        r"$n_{\nu}$=",
        r"$dim_{umap}=$",
        r"$\delta=$",
        r"$n_{s}$=",
        r"$n_{c}$=",
        r"$\epsilon$=",
    ]
    second_line_vals = [
        f"{params['dim_pca']}, ",
        f"{params['umap_n_neighbors']}, ",
        f"{params['umap_n_components']}, ",
        f"{params['umap_min_dist']}, ",
        f"{params['hs_min_sample']}, ",
        f"{params['hs_min_cluster_size']}, ",
        f"{params['hs_cluster_selection_epsilon']}",
    ]
    second_line = "".join(
        [f"{a}{b}" for a, b, in zip(second_line_vars, second_line_vals)]
    )
    axes.format(suptitle=f"{first_line}\n{second_line}")
    if savepath:
        savepath.parent.mkdir(exist_ok=True, parents=True)
        fig.save(savepath)
    pplt.close()
    return None


def calc_cluster_performance(cluster: np.array):
    """Calculater performance measure of clustering results."""
    unique_classes = np.unique(cluster)
    if -1 in unique_classes:
        num_of_classes = len(unique_classes) - 1
    else:
        num_of_classes = len(unique_classes)

    classes_relative = pd.Series(cluster).value_counts(normalize=True).sort_index()

    result_bin = 1
    # if too few or too much classes => bad result
    if num_of_classes < 3 or num_of_classes > 10:
        result_bin = 0

    # if any class contains less than 1% or more than 50% of total points => bad result
    if any(classes_relative < 0.01) or any(classes_relative > 0.5):
        result_bin = 0

    # if noise is larger than 20% of all gridpoints => bad result
    try:
        if classes_relative[-1] > 0.2:
            result_bin = 0
    except KeyError:  # triggers if -1 does not exist => no noise
        pass

    return result_bin


def save_cluster_nc(
    xrda: xr.DataArray, cluster: np.array, params: dict, save_path: Path
):
    """Save cluster with params as attributes to specified path"""
    save_path.parent.mkdir(exist_ok=True, parents=True)
    xy_clust = xrda.isel(variable=0).copy()
    xy_clust["variable"] = "umap_hdbscan"
    xy_clust.values = cluster
    xy_clust.attrs["metric"] = params["metric"]
    xy_clust.attrs["time_period"] = params["time_period"]
    xy_clust.attrs["corr"] = params["corr"]
    xy_clust.attrs["dim_pca"] = params["dim_pca"]
    xy_clust.attrs["umap_n_neighbors"] = params["umap_n_neighbors"]
    xy_clust.attrs["umap_min_dist"] = params["umap_min_dist"]
    xy_clust.attrs["umap_n_components"] = params["umap_n_components"]
    xy_clust.attrs["hs_min_cluster_size"] = params["hs_min_cluster_size"]
    xy_clust.attrs["hs_min_sample"] = params["hs_min_sample"]
    xy_clust.attrs["hs_cluster_selection_epsilon"] = params[
        "hs_cluster_selection_epsilon"
    ]
    xy_clust.attrs["result"] = params["result"]
    xy_clust.attrs["key"] = params["key"]
    xy_clust["pca_explained_variance"] = params["pca_explained_variance"]
    xy_clust.unstack("z").to_netcdf(save_path)
    return None


def save_embedding_csv(embedding: np.array, save_path: Path):
    """Save embedding with params as attributes to specified path"""
    if embedding.shape[1] == 2:
        cols = ["x", "y"]
    elif embedding.shape[1] == 5:
        cols = ["x", "y", "z", "a", "b"]
    elif embedding.shape[1] == 10:
        cols = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"]
    save_path.parent.mkdir(exist_ok=True, parents=True)
    emb_df = pd.DataFrame(
        data=embedding,
        columns=cols,
    )
    emb_df.to_csv(save_path)
    return None


def load_cluster_nc(path_to_file: Path) -> xr.DataArray:
    """Load a previously calculated cluster as xr.DataArray"""
    return xr.load_dataarray(path_to_file)


def get_params_from_xrda_cluster(xrda_clust: xr.DataArray) -> dict:
    """get parameters from xr.DataArray cluster and return
    as dict.
    """
    prms = {}
    prms["key"] = xrda_clust.attrs["key"]
    prms["corr"] = xrda_clust.attrs["corr"]
    prms["dim_pca"] = xrda_clust.attrs["dim_pca"]
    prms["umap_n_neighbors"] = xrda_clust.attrs["umap_n_neighbors"]
    prms["umap_n_components"] = xrda_clust.attrs["umap_n_components"]
    prms["umap_min_dist"] = xrda_clust.attrs["umap_min_dist"]
    prms["hs_min_sample"] = xrda_clust.attrs["hs_min_sample"]
    prms["hs_min_cluster_size"] = xrda_clust.attrs["hs_min_cluster_size"]
    prms["hs_cluster_selection_epsilon"] = xrda_clust.attrs[
        "hs_cluster_selection_epsilon"
    ]
    return prms


def load_embedding_csv(path_to_file: Path) -> pd.DataFrame:
    """Load a previously calculated embedding as pd.DataFrame"""
    return pd.read_csv(path_to_file, index_col=0)


def open_config(experiment_name: str) -> dict:
    """open config file and load params for specific
    experiment into a dict.
    """
    with open(
        Path(Path(__file__).parent, "config_experiments.json"), encoding="utf-8"
    ) as cfg:
        cfg_dict = json.load(cfg)[experiment_name]
    return cfg_dict


ordered_variable_list = [
    # temperature: n=80
    "GSL",
    "Tyearmean",
    "TNyearmean",
    "FD",
    "TR",
    "SU",
    "HD",
    "ID",
    "DTR",
    "bio1",
    "bio2",
    "bio3",
    "bio4",
    "bio5",
    "bio6",
    "bio7",
    "bio10",
    "bio11",
    "ET0_yearmean",
    "ET0_yearstd",
    "ETR_01",
    "ETR_02",
    "ETR_03",
    "ETR_04",
    "ETR_05",
    "ETR_06",
    "ETR_07",
    "ETR_08",
    "ETR_09",
    "ETR_10",
    "ETR_11",
    "ETR_12",
    "TNn_01",
    "TNn_02",
    "TNn_03",
    "TNn_04",
    "TNn_05",
    "TNn_06",
    "TNn_07",
    "TNn_08",
    "TNn_09",
    "TNn_10",
    "TNn_11",
    "TNn_12",
    "TNx_01",
    "TNx_02",
    "TNx_03",
    "TNx_04",
    "TNx_05",
    "TNx_06",
    "TNx_07",
    "TNx_08",
    "TNx_09",
    "TNx_10",
    "TNx_11",
    "TNx_12",
    "TXn_01",
    "TXn_02",
    "TXn_03",
    "TXn_04",
    "TXn_05",
    "TXn_06",
    "TXn_07",
    "TXn_08",
    "TXn_09",
    "TXn_10",
    "TXn_11",
    "TXn_12",
    "TXx_01",
    "TXx_02",
    "TXx_03",
    "TXx_04",
    "TXx_05",
    "TXx_06",
    "TXx_07",
    "TXx_08",
    "TXx_09",
    "TXx_10",
    "TXx_11",
    "TXx_12",
    # precipitation: n=20
    "CDD",
    "CWD",
    "PRCPTOT",
    "R10mm",
    "R20mm",
    "RRRx1day",
    "RRyearmean",
    "Rx5day",
    "SDII",
    "api_p0.935_k30_yearmean",
    "api_p0.935_k30_yearstd",
    "api_p0.935_k7_yearmean",
    "api_p0.935_k7_yearstd",
    "pci",
    "bio12",
    "bio13",
    "bio14",
    "bio15",
    "bio16",
    "bio17",
    # radiation: n=4
    "SA_yearmean",
    "SA_yearstd",
    "SR_yearmean",
    "SR_yearstd",
    # snow: n=13
    "snow_depth_gt_1",
    "snow_depth_yearmean",
    "snow_depth_yearstd",
    "snow_depth_diff_count_gt_0.02",
    "snow_depth_diff_count_lt_-0.03",
    "snow_depth_diff_yearmean",
    "snow_depth_diff_yearstd",
    "swe_tot_yearmean",
    "swe_tot_yearstd",
    "swe_tot_diff_count_gt_3",
    "swe_tot_diff_count_lt_-5",
    "swe_tot_diff_yearmean",
    "swe_tot_diff_yearstd",
    # combined: n=10
    "bio8",
    "bio9",
    "bio18",
    "bio19",
    "SPEI30_yearmean",
    "SPEI30_yearstd",
    "SPEI90_yearmean",
    "SPEI90_yearstd",
    "SPEI365_yearmean",
    "SPEI365_yearstd",
    # terrain: n=49
    "dtm_NTO_average",
    "dtm_NTO_rms",
    "dtm_curv-max_average",
    "dtm_curv-max_rms",
    "dtm_TWI_average",
    "dtm_TWI_rms",
    "dtm_SCA_average",
    "dtm_SCA_rms",
    "dtm_catchment-area_average",
    "dtm_catchment-area_rms",
    "dtm_PTO_average",
    "dtm_PTO_rms",
    "dtm_SVF_average",
    "dtm_SVF_rms",
    "dtm_maximum-height_average",
    "dtm_maximum-height_rms",
    "dtm_average",
    "dtm_rms",
    "dtm_TPI_average",
    "dtm_TPI_rms",
    "dtm_VRM_average",
    "dtm_VRM_rms",
    "dtm_curv-prof_average",
    "dtm_convexity_average",
    "dtm_convexity_rms",
    "dtm_curv-min_average",
    "dtm_curv-plan_average",
    "dtm_DAH_average",
    "dtm_slope-rad_average",
    "dtm_DAH_rms",
    "dtm_MRN_average",
    "dtm_MRN_rms",
    "dtm_TRI_average",
    "dtm_TRI_rms",
    "dtm_roughness_average",
    "dtm_roughness_rms",
    "dtm_slope-rad_rms",
    "dtm_slope_average",
    "dtm_slope_rms",
    "dtm_convergence-index_average",
    "dtm_SPI_average",
    "dtm_SPI_rms",
    "dtm_geomorphons_mode",
    "dtm_convergence-index_rms",
    "dtm_curv-min_rms",
    "dtm_curv-prof_rms",
    "dtm_curv-plan_rms",
    "dtm_aspect-cos_sum",
    "dtm_aspect-sin_sum",
]
