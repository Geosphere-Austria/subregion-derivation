"""Explorative analysis of cluster distance metrics and clustering scores"""

import xarray as xr
import rioxarray
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, scale
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance as wd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# load variable per clusters
clusters_ml_path = "dat/out/physioclimatic_clusters_raster_AT.tif"
features_path = "dat/interim/07_cluster_evaluation/core_variables_r0.8.nc"
base_params_path = "dat/interim/07_cluster_evaluation/base_variables.nc"
clusters_ml = rioxarray.open_rasterio(clusters_ml_path).squeeze()
clusters_ml = clusters_ml.where(clusters_ml != -9999, np.nan)
clusters_ml = clusters_ml.drop(["band", "spatial_ref"])
features = xr.open_dataarray(features_path)
base_params = xr.open_dataarray(base_params_path)

# standardize input features
features_norm = features.stack(z=["y", "x"]).transpose("z", "variable")
var_coord = features_norm["variable"]
z_coord = features_norm.dropna(dim="z").z
features_dims = features_norm.dims
norm_raw = scale(features_norm.dropna(dim="z"))
features_norm = xr.DataArray(
    data=norm_raw, coords=[z_coord, var_coord], dims=features_dims
)


def kmeans(input_: np.array, n_clusters: int = 7):
    # same number of clusters as above
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans_clusters = kmeans.fit_predict(input_)
    print(f"{kmeans_clusters.shape = }")  # (spatial)
    print(
        f"{np.unique(kmeans_clusters) = }"
    )  # by default starts with 0, add 1 such that it's similar to above
    kmeans_clusters = kmeans_clusters + 1
    print(
        f"{np.unique(kmeans_clusters) = }"
    )  # by default starts with 0, add 1 such that it's similar to above
    return kmeans_clusters


def cluster_to_xda(data, coord, dim):
    return xr.DataArray(
        data=data,
        coords=[
            coord,
        ],
        dims=[
            dim,
        ],
    )


dim_pca = 20
pca = PCA(n_components=dim_pca)  # similar PCA dimension compared to clustering from ml
features_pca = pca.fit(features_norm.T).components_.T
expl_var = pca.fit(features_norm.T).explained_variance_ratio_
# save principal components
pca_coords = [f"pca_{num+1}" for num in range(dim_pca)]
principal_components = xr.DataArray(
    data=features_pca, coords=[features_norm.z, pca_coords], dims=["z", "pca_dim"]
).unstack("z")
principal_components.name = "principal components"
principal_components = principal_components.isel(y=slice(None, None, -1)).to_dataset()
expl_var = xr.DataArray(
    data=expl_var,
    coords=[
        pca_coords,
    ],
    dims=[
        "pca_dim",
    ],
)
expl_var.name = "explained variance"
principal_components = principal_components.assign(explained_variance=expl_var)
principal_components.to_netcdf(
    "dat/interim/07_cluster_evaluation/principal_components_dim20.nc"
)
print(f"{features_pca.shape = }")  # (spatialm dim_pca)
# repeat same steps from before
clusters_km = kmeans(input_=features_pca, n_clusters=7)
clusters_km = cluster_to_xda(clusters_km, coord=z_coord, dim="z").unstack("z")
clusters_km.rio.to_raster("dat/out/pca_kmeans_clusters_raster_at.tif")


# load variable per clusters
unique_clusters = np.unique(clusters_ml)[np.where(np.unique(clusters_ml) > 0)[0]]
cluster_combinations = [(i, j) for i, j in combinations(unique_clusters, 2)]

base_params_df = pd.read_csv(
    "dat/interim/07_cluster_evaluation/base_variables.csv", index_col=0
)
variables_dict = {
    "RR": base_params_df[["y", "x", "RR"]],
    "T": base_params_df[["y", "x", "T"]],
    "slope": base_params_df[["y", "x", "slope"]],
}


def cluster_da_to_df(da_in: xr.DataArray, name: str) -> pd.DataFrame:
    """transform cluster data from xr.DataArray
    to pd.DataFrame
    """
    df_out = da_in.to_dataframe(name=name).reset_index().dropna().astype(int)
    return df_out


clusters_ml_df = cluster_da_to_df(da_in=clusters_ml, name="umap_hdbscan")
clusters_km_df = cluster_da_to_df(da_in=clusters_km, name="kmeans")

clusters_params_df = pd.concat(
    [
        base_params_df.set_index(["y", "x"]),
        clusters_ml_df.set_index(["y", "x"]),
        clusters_km_df.set_index(["y", "x"]),
    ],
    axis=1,
)

clusters_ints = [1, 2, 3, 4, 5, 6, 7]
clust_comb = [c for c in combinations(clusters_ints, 2)]

clusters_ml_data = clusters_ml.stack(z=["y", "x"]).dropna(dim="z").values
clusters_km_data = clusters_km.stack(z=["y", "x"]).dropna(dim="z").values
base_params_data = (
    base_params.stack(z=["y", "x"]).dropna(dim="z").transpose("z", ...).values
)
print(f"{clusters_ml_data.shape = }")
print(f"{clusters_km_data.shape = }")
print(f"{features_pca.shape = }")
print(f"{base_params_data.shape = }")


def transform_data_np(meteo_values: np.array, transform: str = None) -> np.array:
    # By default, zero-mean, unit-variance normalization is applied to the transformed data.
    assert transform in [None, "yeo-johnson", "box-cox"]
    if transform:
        yj = PowerTransformer(method=transform)
        data_tf = yj.fit_transform(meteo_values)
    return data_tf


def calc_wasserstein_score(data_input, cluster_classes):
    ws_score = 0
    unique_clusters = np.unique(cluster_classes)
    var_num = data_input.shape[1]
    wd_list = []
    for class1, class2 in combinations(unique_clusters, 2):
        data1 = data_input[cluster_dat == class1, :]
        data2 = data_input[cluster_dat == class2, :]
        for iii in range(var_num):
            data_iter1 = data1[:, iii]
            data_iter2 = data2[:, iii]
            wd_list.append(wd(data_iter1, data_iter2))
    ws_score = np.mean(wd_list)
    return ws_score


def calc_score(data_in, cluster_truth, score_func, metric):
    if score_func == "silhouette":
        if not metric:
            score_ = silhouette_score(data_in, cluster_truth)
        else:
            score_ = silhouette_score(data_in, cluster_truth, metric=metric)
    elif score_func == "davies_bouldin":
        score_ = davies_bouldin_score(data_in, cluster_truth)
    elif score_func == "calinski_harabasz":
        score_ = calinski_harabasz_score(data_in, cluster_truth)
    return score_


eval_metrics = [
    "silhouette;features",
    "silhouette;base_params",
    "silhouette-seuclidean;features",
    "silhouette-seuclidean;base_params",
    "silhouette-mahalanobis;features",
    "silhouette-mahalanobis;base_params",
    "davies_bouldin;features",
    "davies_bouldin;base_params",
    "calinski_harabasz;features",
    "calinski_harabasz;base_params",
]
eval_clusters = ["umap_hdbscan", "kmeans"]
results = pd.DataFrame(np.nan, columns=eval_clusters, index=eval_metrics)

# yeo-johnson transform of features
# base-param_transform
base_params_transform = transform_data_np(
    meteo_values=base_params_data, transform="yeo-johnson"
)


eval_ml_cluster = clusters_ml_data[clusters_ml_data != -1]
eval_ml_features = features_pca[clusters_ml_data != -1]
eval_ml_base = base_params_data[clusters_ml_data != -1]
eval_km_cluster = clusters_km_data
eval_km_features = features_pca
eval_km_base = base_params_data

for cluster_, cluster_dat in zip(eval_clusters, [eval_ml_cluster, eval_km_cluster]):
    print(f"--working on {cluster_}--")
    for eval_ in eval_metrics:
        print(f"--working on {eval_}--")
        score, input_ = eval_.split(";")
        if score.startswith("silhouette-"):
            score, metric = score.split("-")
        else:
            metric = None
        if input_ == "features":
            if cluster_ == "umap_hdbscan":
                in_ = eval_ml_features
            elif cluster_ == "kmeans":
                in_ = eval_km_features
        elif input_ == "base_params":
            if cluster_ == "umap_hdbscan":
                in_ = eval_ml_base
            elif cluster_ == "kmeans":
                in_ = eval_km_base
        results.loc[eval_, cluster_] = calc_score(in_, cluster_dat, score, metric)

eval_metrics_wd = ["wasserstein_distance;features", "wasserstein_distance;base_params"]
results_wd = pd.DataFrame(np.nan, columns=eval_clusters, index=eval_metrics_wd)

for cluster_, cluster_dat in zip(eval_clusters, [eval_ml_cluster, eval_km_cluster]):
    print(f"--working on {cluster_}--")
    for eval_ in eval_metrics_wd:
        score, input_ = eval_.split(";")
        if input_ == "features":
            if cluster_ == "umap_hdbscan":
                in_ = eval_ml_features
            elif cluster_ == "kmeans":
                in_ = eval_km_features
        elif input_ == "base_params":
            if cluster_ == "umap_hdbscan":
                in_ = eval_ml_base
            elif cluster_ == "kmeans":
                in_ = eval_km_base
        print(f"{in_.shape = }")
        print(f"{cluster_dat.shape = }")
        in_standardized = transform_data_np(meteo_values=in_, transform="yeo-johnson")
        results_wd.loc[eval_, cluster_] = calc_wasserstein_score(
            in_standardized, cluster_dat
        )

results_total = pd.concat([results, results_wd], axis=0)

methods_str = [
    row[0].split(";") for row in results_total.reset_index()[["index"]].values
]
methods_str_df = pd.DataFrame(methods_str, columns=["metric", "input_variables"])

ml_vals = results_total.reset_index()[["umap_hdbscan"]]
km_vals = results_total.reset_index()[["kmeans"]]
ml_vals_df = pd.concat([ml_vals, methods_str_df], axis=1).rename(
    {"umap_hdbscan": "score"}, axis=1
)
ml_vals_df["cluster_method"] = "umap_hdbscan"
km_vals_df = pd.concat([km_vals, methods_str_df], axis=1).rename(
    {"kmeans": "score"}, axis=1
)
km_vals_df["cluster_method"] = "kmeans"
results_df = pd.concat([ml_vals_df, km_vals_df], axis=0)
results_df


def set_layout(ax, layout_style):
    if layout_style == "silhouette":
        ylimhi = 1
        ylimlo = -1
        reverse = False
        histartanchor = ylimhi * 0.5
        lostartanchor = ylimlo * 0.5
    elif layout_style == "silhouette-seuclidean":
        ylimhi = 1
        ylimlo = -1
        reverse = False
        histartanchor = ylimhi * 0.5
        lostartanchor = ylimlo * 0.5
    elif layout_style == "silhouette-mahalanobis":
        ylimhi = 1
        ylimlo = -1
        reverse = False
        histartanchor = ylimhi * 0.5
        lostartanchor = ylimlo * 0.5
    elif layout_style == "davies_bouldin":
        ylimhi = None
        ylimlo = None
        reverse = True
        _, hitmp = ax.get_ylim()
        histartanchor = hitmp * 0.25
        lostartanchor = hitmp * 0.75
    elif layout_style == "calinski_harabasz":
        ylimhi = None
        ylimlo = 0
        reverse = False
        _, hitmp = ax.get_ylim()
        histartanchor = hitmp * 0.75
        lostartanchor = hitmp * 0.25
    elif layout_style == "wasserstein_distance":
        ylimhi = None
        ylimlo = 0
        reverse = False
        _, hitmp = ax.get_ylim()
        histartanchor = hitmp * 0.75
        lostartanchor = hitmp * 0.25

    # silhouette: higher = better; score around 0 indicates overlapping clusters,
    # silhouette: score=[-1,1]
    # davies_bouldin: best score = 0; score=[0,inf)
    # calinski_harabasz: higher = better, score=[0,inf)
    # wasserstein: higher = better
    ax.set_ylim([ylimlo, ylimhi])
    lo, hi = ax.get_ylim()
    if reverse:
        ax.invert_yaxis()
        reverse_scale = -1
    else:
        reverse_scale = 1
    histart = (hi + lo) / 2 + ((hi - lo) * 0.025 * reverse_scale)
    lostart = (hi + lo) / 2 - ((hi - lo) * 0.025 * reverse_scale)
    ax.annotate(
        "better",
        xy=(0.5, histartanchor),
        xytext=(0.5, histart),
        arrowprops=dict(width=5, headwidth=15, headlength=15, color="k"),
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.annotate(
        "worse",
        xy=(0.5, lostartanchor),
        xytext=(0.5, lostart),
        arrowprops=dict(width=5, headwidth=15, headlength=15, color="k"),
        horizontalalignment="center",
        verticalalignment="center",
    )
    return None


# fig, _ = plt.subplots(figsize=(15,10))
# ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
# ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
# ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
# ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
_, axes = plt.subplots(ncols=3, nrows=2, figsize=(13, 8), sharex=True)
# axesiter = iter(axes.ravel())
# for metric, ax in zip(results_df.metric.unique(), [ax1, ax2, ax3, ax4, ax5]):
for metric, ax in zip(results_df.metric.unique(), iter(axes.ravel())):
    ax.axhline(0, color="k", alpha=0.5, ls="--", lw=0.7)
    sns.barplot(
        results_df.query(f"metric == '{metric}'"),
        x="input_variables",
        y="score",
        hue="cluster_method",
        width=0.6,
        ax=ax,
        legend=False,
    )
    ax.set_title(f"{metric} score")
    set_layout(ax=ax, layout_style=metric)
# plt.subplots_adjust(hspace=0.2, wspace=0.8)
plt.subplots_adjust(hspace=0.1, wspace=0.4)

import matplotlib.patches as mpatches

leg1 = mpatches.Patch(color=sns.color_palette("deep")[0], label="umap_hdbscan")
leg2 = mpatches.Patch(color=sns.color_palette("deep")[1], label="kmeans")
# ax.legend(handles=[leg1, leg2], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., title="Clustering method")
ax.legend(
    handles=[leg1, leg2],
    bbox_to_anchor=(-1.43, 2.4),
    loc="upper left",
    borderaxespad=0.0,
    title="Clustering method",
    ncols=2,
)
plt.savefig("plt/cluster_evaluation.png", dpi=300)
