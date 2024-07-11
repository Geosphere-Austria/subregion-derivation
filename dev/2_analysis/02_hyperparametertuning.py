"""modelling of subregions with various clustering approaches"""
from pathlib import Path
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

from ..utils import (
    apply_hdbscan,
    apply_pca,
    apply_umap,
    load_clean_data,
    normalize_data,
    open_config,
    plot_composit_umap_hdbscan,
    calc_cluster_performance,
    save_cluster_nc,
    save_embedding_csv,
)


if __name__ == "__main__":
    # define experiment name
    exp_dict = {
        "1": "experiment_umap_hyperparams",
        "2": "experiment_umap_hyperparams2",
        "3": "experiment_hdbscan_hyperparams",
        "4": "experiment_umap_param_showcase",
        "5": "experiment_hdbscan_param_showcase",
    }
    print("Which experiment shall be run:")
    print(
        "1: experiment_umap_hyperparams\n2: experiment_umap_hyperparams2"
        "\n3: experiment_hdbscan_hyperparams\n4: experiment_umap_param_showcase"
        "\n5: experiment_hdbscan_param_showcase"
    )
    exp_name = exp_dict[input("Enter number (1..5): ")]
    print(f"Running experiment '{exp_name}'")

    filter_ = True

    if filter_:
        str_suffix = ""
    else:
        str_suffix = "_unfiltered"

    config = open_config(experiment_name=exp_name)

    # define fixed configs params
    metrics = [
        "mean",
    ]
    time_periods = [
        1,
    ]
    corr_range = [0.7, 0.8, 0.9, 0.95]

    # define variable config params
    base_path = Path(f"dat/interim/05_hyperparametertuning/{exp_name}")
    dims_pca = config["dims_pca"]
    umap_n_neighbors_list = config["umap_n_neighbors_list"]
    umap_min_dist_list = config["umap_min_dist_list"]
    umap_n_components_list = config["umap_n_components_list"]
    hs_min_cluster_size_list = config["hs_min_cluster_size_list"]
    hs_min_samples_list = config["hs_min_samples_list"]
    hs_cluster_selection_epsilon_list = config["hs_cluster_selection_epsilon_list"]

    df_list = []
    for (
        metric,
        time_period,
        corr,
        dim_pca,
    ) in product(metrics, time_periods, corr_range, dims_pca):
        params = {
            "time_period": time_period,
            "corr": corr,
            "dim_pca": dim_pca,
            "metric": metric,
        }

        intermediate_path = Path(f"t{time_period}_r{corr}_npca{dim_pca}")
        variables_file = lambda t, r: Path(
            f"dat/interim/03_feature_definition{str_suffix}/core_variables_{metric}_t{t}_r{r}.nc"
        )
        dat_raw = load_clean_data(
            corrcoef=corr,
            time_period=time_period,
            metric=metric,
            path_to_file=variables_file(time_period, corr),
        )

        # if dim data < dim_pca, skip
        dim_data = dat_raw["variable"].shape[0]
        if dim_pca > dim_data:
            print(f"..{dim_pca = } larger than {dim_data = }, skipping..")
            continue

        dat_norm = normalize_data(data=dat_raw)
        # pca
        if dim_pca:
            dat_in, pca_explained_variance = apply_pca(data=dat_norm, n_pca=dim_pca)
        else:
            dat_in = dat_norm
            explained_variance = None

        params["pca_explained_variance"] = pca_explained_variance

        # umap
        for umap_n_neighbors, umap_min_dist, umap_n_components in product(
            umap_n_neighbors_list,
            umap_min_dist_list,
            umap_n_components_list,
        ):
            params["umap_n_neighbors"] = umap_n_neighbors
            params["umap_min_dist"] = umap_min_dist
            params["umap_n_components"] = umap_n_components
            embedding = apply_umap(data=dat_in, params=params)

            umap_path = Path(
                f"neighbours{umap_n_neighbors}_mindist{umap_min_dist}_dim{umap_n_components}"
            )

            for (
                hs_min_cluster_size,
                hs_min_samples,
                hs_cluster_selection_epsilon,
            ) in product(
                hs_min_cluster_size_list,
                hs_min_samples_list,
                hs_cluster_selection_epsilon_list,
            ):
                if hs_min_samples > hs_min_cluster_size:
                    continue
                params["hs_min_cluster_size"] = hs_min_cluster_size
                params["hs_min_sample"] = hs_min_samples
                params["hs_cluster_selection_epsilon"] = hs_cluster_selection_epsilon
                clust = apply_hdbscan(data=embedding, params=params)

                result = calc_cluster_performance(cluster=clust)
                params["result"] = result

                key = int(np.random.rand(1) * 100_000_000)
                params["key"] = key
                timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
                print(f"{timestamp}..finished cluster for {key = }")

                composite_path = Path(base_path, intermediate_path, umap_path)

                # if result == 1:
                base_path.mkdir(exist_ok=True, parents=True)
                fig_savepath = Path(
                    composite_path,
                    f"fig/composite_umap_hdbscan_cs_{hs_min_cluster_size}_eps_{hs_cluster_selection_epsilon}_{key}.png",
                )
                plot_composit_umap_hdbscan(
                    clust=clust,
                    dat_raw=dat_raw,
                    embedding=embedding,
                    params=params,
                    savepath=fig_savepath,
                )

                embedding_savepath = Path(
                    composite_path, f"dat_csv/embedding_out_{key}.nc"
                )
                save_embedding_csv(embedding=embedding, save_path=embedding_savepath)
                dat_savepath = Path(composite_path, f"dat_nc/cluster_out_{key}.nc")
                save_cluster_nc(
                    xrda=dat_raw, cluster=clust, params=params, save_path=dat_savepath
                )

                df = pd.DataFrame.from_dict(params, orient="index").T
                df["result"] = result
                df["key"] = key

                unique_classes = np.unique(clust)
                if -1 in unique_classes:
                    num_of_classes = len(unique_classes) - 1
                else:
                    num_of_classes = len(unique_classes)
                df["num_of_classes"] = num_of_classes

                results_path = Path(composite_path, "dat_csv")
                results_path.mkdir(exist_ok=True, parents=True)
                df.to_csv(Path(results_path, f"results_{key}.csv"))
                df_list.append(df)

    pd.concat(df_list).reset_index().drop("index", axis=1).to_csv(
        Path(base_path, "results.csv")
    )
