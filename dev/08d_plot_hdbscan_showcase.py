"""plotting umap showcase for different parameters"""
from pathlib import Path

import pandas as pd
import numpy as np
import proplot as pplt

from utils import load_embedding_csv, load_cluster_nc, get_params_from_xrda_cluster

exp_path = Path(
    "dat/interim/" "05_hyperparametertuning/experiment_hdbscan_param_showcase"
)

hdbscan_params = {
    "500": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
    "1000": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
    "2000": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
    "3000": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
    "4000": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
    "5000": {
        "0.1": "",
        "0.5": "",
        "0.8": "",
        "1": "",
        "2": "",
        "5": "",
    },
}
## load all embeddings
for file in exp_path.rglob("embedding*.nc"):
    cluster_file = Path(
        str(file).replace("embedding", "cluster").replace("/dat_csv/", "/dat_nc/")
    )
    embedding = np.array(load_embedding_csv(file))
    xrda_clust = load_cluster_nc(cluster_file)
    xrda_clust = xrda_clust.stack(z=["x", "y"]).dropna(dim="z")
    params = get_params_from_xrda_cluster(xrda_clust)
    hdbscan_params[str(params["hs_min_cluster_size"])][
        str(params["hs_cluster_selection_epsilon"])
    ] = (embedding, xrda_clust)

# prepare colours
noise = (0.4, 0.4, 0.4)
alpen = (0.27, 0.42, 0.65)
inneralpine = (0.80, 0.73, 0.46)
sued_stmk_area = (0.36, 0.31, 0.56)
sued_stmk_tal = (0.67, 0.61, 0.87)
sued_stmk = (0.51, 0.45, 0.71)
sued_stmk_rot = (0.77, 0.31, 0.32)
eastern_low = (0.87, 0.52, 0.32)
eastern_basin = (0.39, 0.71, 0.81)
innviertel = (0.86, 0.55, 0.77)
muehl_waldviertel = (0.33, 0.66, 0.40)
donau = (0.58, 0.47, 0.38)
gray = (0.5, 0.5, 0.5)
colors_dict = {
    -1: noise,
    0: muehl_waldviertel,
    1: sued_stmk_rot,
    2: alpen,
    3: sued_stmk_area,
    4: eastern_basin,
    5: inneralpine,
    6: sued_stmk_tal,
    7: eastern_low,
    8: innviertel,
    9: donau,
    10: gray,
}

# plot
fig, axes = pplt.subplots(
    ncols=6,
    nrows=6,
    sharex=True,
    sharey=True,
    figsize=(15, 15),
    space=0,
    span=False,
    toplabels=("0.1", "0.5", "0.8", "1", "2", "5"),
    leftlabels=("500", "1000", "2000", "3000", "4000", "5000"),
)

ax_iter = iter(axes)
axes.format(
    suptitle=r"selection epsilon $\epsilon$",
    xlocator=5,
    ylocator=5,
    xlim=(-4, 18),
    ylim=(-7, 15),
)
j = 0
for neigh, min_dist_dict in hdbscan_params.items():
    for min_d, (emb, clus) in min_dist_dict.items():
        ax = next(ax_iter)

        large_clusters = (
            pd.Series(clus)
            .value_counts()
            .sort_index()[pd.Series(clus).value_counts().sort_index() > 1000]
        )
        if -1 in large_clusters.keys():
            large_clusters = large_clusters.drop(-1)
        old_index = large_clusters.sort_values(ascending=False).index
        large_clusters = (
            large_clusters.sort_values(ascending=False)
            .reset_index()
            .drop("index", axis=1)
        )
        remap_dict = {
            key: value for key, value in zip(old_index, range(old_index.shape[0]))
        }
        remapped_clust = np.copy(clus)
        for key, value in remap_dict.items():
            remapped_clust[clus == key] = value
        len_of_large_clusters = len(large_clusters)
        c_dict = {}
        for key_clr, clr in colors_dict.items():
            c_dict[key_clr] = clr
        colors = [c_dict[x] for x in remapped_clust]

        ax.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.1, s=2)
        if j % 6 != 0:
            ax.format(ytickloc="neither")
        j += 1

fig.text(-0.004, 0.5, r"min_cluster_size $n_c$", fontweight="bold", rotation=90)
fig.save("plt/fig_hdbscan_param_showcase_pbf.png", dpi=300, bbox_inches="tight")
