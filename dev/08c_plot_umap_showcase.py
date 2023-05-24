"""plotting umap showcase for different parameters"""
from pathlib import Path

import pandas as pd
import numpy as np
import proplot as pplt

from utils import load_embedding_csv, load_cluster_nc, get_params_from_xrda_cluster

colorscale = input("chose color scale ('color' or 'gray'): ")

if colorscale == "color":
    print("using normal color scale")
elif colorscale == "gray":
    print("using gray color scale")
else:
    raise ValueError(f"{colorscale = } not valid; use 'color', or 'gray'")

xrda_clust = load_cluster_nc(  # load specific cluster result for colouring
    Path(
        "dat/interim/05_hyperparametertuning/experiment_hdbscan_hyperparams"
        "/t1_r0.8_npca20/neighbours50_mindist0.01_dim2"
        "/dat_nc/cluster_out_2792936.nc"
    )
)
xrda_clust = xrda_clust.stack(z=["x", "y"]).dropna(dim="z")

exp_path = Path(
    "dat/interim/"
    "05_hyperparametertuning/experiment_umap_param_showcase/"
    "t1_r0.8_npca20/"
)

umap_params = {
    "10": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
    "20": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
    "30": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
    "50": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
    "100": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
    "150": {"0.01": "", "0.05": "", "0.1": "", "0.3": "", "0.5": "", "1": ""},
}
## load all embeddings
for file in exp_path.rglob("embedding*.nc"):
    cluster_file = list(file.parents[1].rglob("cluster_out*.nc"))[0]
    embedding = np.array(load_embedding_csv(file))
    params = get_params_from_xrda_cluster(load_cluster_nc(cluster_file))
    umap_params[str(params["umap_n_neighbors"])][
        str(params["umap_min_dist"])
    ] = embedding

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
colors_dict = {
    -1: noise,
    0: muehl_waldviertel,
    1: sued_stmk_rot,
    2: alpen,
    3: sued_stmk_area,
    4: eastern_basin,
    5: inneralpine,
    6: sued_stmk_tal,
}
large_clusters = (
    pd.Series(xrda_clust)
    .value_counts()
    .sort_index()[pd.Series(xrda_clust).value_counts().sort_index() > 1000]
)
if -1 in large_clusters.keys():
    large_clusters = large_clusters.drop(-1)
old_index = large_clusters.sort_values(ascending=False).index
large_clusters = (
    large_clusters.sort_values(ascending=False).reset_index().drop("index", axis=1)
)
remap_dict = {key: value for key, value in zip(old_index, range(old_index.shape[0]))}
remapped_clust = np.copy(xrda_clust)
for key, value in remap_dict.items():
    remapped_clust[xrda_clust == key] = value

if colorscale == "color":
    c_dict = {}
    for key_clr, clr in colors_dict.items():
        c_dict[key_clr] = clr
    colors = [c_dict[x] for x in remapped_clust]
    cscale_str = ""
elif colorscale == "gray":
    colors = "gray"
    cscale_str = "_gray"

# plot
fig, axes = pplt.subplots(
    ncols=6,
    nrows=6,
    sharex=True,
    sharey=True,
    figsize=(15, 15),
    space=0,
    span=False,
    toplabels=("0.01", "0.05", "0.1", "0.3", "0.5", "1"),
    leftlabels=("10", "20", "30", "50", "100", "150"),
)
ax_iter = iter(axes)
axes.format(
    suptitle=r"min_dist $\delta$",
    xlocator=5,
    ylocator=5,
    xlim=(-12, 21),
    ylim=(-10, 25),
)
j = 0
for neigh, min_dist_dict in umap_params.items():
    for min_d, emb in min_dist_dict.items():
        ax = next(ax_iter)
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, alpha=0.1, s=2)
        if j % 6 != 0:
            ax.format(ytickloc="neither")
        j += 1

fig.text(-0.004, 0.5, r"n_neighbours $n_\nu$", fontweight="bold", rotation=90)
fig.save(
    f"plt/fig_umap_param_showcase_pbf{cscale_str}.png", dpi=300, bbox_inches="tight"
)
