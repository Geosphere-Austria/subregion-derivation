"""plotting routine for composite cluster figures"""

from pathlib import Path

import numpy as np
import proplot as pplt
import pandas as pd
import seaborn as sns

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    plot_composit_umap_hdbscan,
    load_embedding_csv,
    load_cluster_nc,
    get_params_from_xrda_cluster,
)

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


def create_final_plot(base_path, key, clrd_, figpath):
    """convenience function to generate finalized plots"""
    embedding = np.array(
        load_embedding_csv(Path(base_path, f"dat_csv/embedding_out_{key}.nc"))
    )
    xrda_clust = load_cluster_nc(Path(base_path, f"dat_nc/cluster_out_{key}.nc"))

    xrda_clust = xrda_clust.stack(z=["x", "y"]).dropna(dim="z")
    params_ = get_params_from_xrda_cluster(xrda_clust)

    plot_composit_umap_hdbscan(
        params=params_,
        dat_raw=xrda_clust,
        embedding=embedding,
        clust=xrda_clust,
        savepath=figpath,
        colors_dict=clrd_,
    )
    return None


####################################
####################################
#### plot 1: 7 clusters

base_path_in = Path(
    "dat/interim/05_hyperparametertuning/experiment_hdbscan_hyperparams"
    "/t1_r0.8_npca20/neighbours50_mindist0.01_dim2"
)
key_in = "2792936"
clrd_in = {
    -1: noise,
    0: muehl_waldviertel,
    1: sued_stmk_rot,
    2: alpen,
    3: sued_stmk_area,
    4: eastern_basin,
    5: inneralpine,
    6: sued_stmk_tal,
}
figpath_in = Path("plt/fig_results_7clust.png")
create_final_plot(base_path=base_path_in, key=key_in, clrd_=clrd_in, figpath=figpath_in)

####################################
####################################
#### plot 2: 4 clusters

base_path_in = Path(
    "dat/interim/05_hyperparametertuning/experiment_hdbscan_hyperparams"
    "/t1_r0.8_npca20/neighbours50_mindist0.01_dim2"
)
key_in = "50143028"
clrd_in = {
    -1: noise,
    0: sued_stmk_rot,
    1: muehl_waldviertel,
    2: alpen,
    3: eastern_basin,
}
figpath_in = Path("plt/fig_results_4clust.png")
create_final_plot(base_path=base_path_in, key=key_in, clrd_=clrd_in, figpath=figpath_in)
