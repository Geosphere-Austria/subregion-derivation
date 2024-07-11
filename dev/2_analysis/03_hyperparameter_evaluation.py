"""evaluate the results of hyperparameter tuning"""
from pathlib import Path
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd
import proplot as pplt
import xarray as xr


def group_df_min_median_max(
    df_in: pd.DataFrame, groups: List[str]
) -> Tuple[pd.DataFrame, int]:
    """group input df by supplied list of str and calculate min, median, max"""
    group_median = df_in.groupby(groups).median()
    group_min = df_in.groupby(groups).min()
    group_max = df_in.groupby(groups).max()

    new_index = [
        "_".join(
            [f"{name}_{entry}" for entry, name in zip(row, group_median.index.names)]
        )
        for row in group_median.index
    ]
    group_median.index = new_index
    group_min.index = new_index
    group_max.index = new_index

    group_median.columns = ["median_classes"]
    group_min.columns = ["min_classes"]
    group_max.columns = ["max_classes"]

    concat = pd.concat([group_min, group_median, group_max], axis=1)
    group_len = [group for group in df_in.groupby(groups)][-1][1].shape[0]
    return concat, group_len


def get_groups(experiment: str) -> List[str]:
    """return list of groups based on experiment"""
    group_umap = ["umap_n_neighbors", "umap_min_dist", "umap_n_components"]
    group_hdbscan = [
        "hs_min_cluster_size",
        "hs_min_sample",
        "hs_cluster_selection_epsilon",
    ]
    if (experiment == "experiment_umap_hyperparams") or (
        experiment == "experiment_umap_hyperparams2"
    ):
        return group_umap, group_hdbscan
    elif experiment == "experiment_hdbscan_hyperparams":
        return group_hdbscan, group_umap
    else:
        raise ValueError(f"invalid {experiment = }")


if __name__ == "__main__":
    # define experiment name
    print("Which experiment shall be evaluated:")
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
    print(f"Evaluating experiment '{exp_name}'")

    eval_type = input("Evaluation Type (1: table 2: heatmaps 3: both): ")

    str_suffix = ""

    parent_dir = Path(f"dat/interim/05_hyperparametertuning")
    results_path = Path(parent_dir, exp_name, "results.csv")
    df_results = pd.read_csv(results_path, index_col=0)
    df_good = (
        df_results.query("result == 1")
        .drop(["metric", "time_period", "pca_explained_variance"], axis=1)
        .copy()
    )
    drop_cols = ["metric", "pca_explained_variance", "key"]  # "result",
    df_results = df_results.drop(drop_cols, axis=1)

    if (eval_type == "1") or (eval_type == "3"):
        ## parse results.csv and save as results_good.csv
        # check noise level
        group_cols = [
            "corr",
            "dim_pca",
            "umap_n_neighbors",
            "umap_min_dist",
            "umap_n_components",
            "hs_min_cluster_size",
            "hs_min_sample",
            "hs_cluster_selection_epsilon",
        ]
        new_results_list = []
        for idx, group in df_good.groupby(group_cols):
            (
                corr,
                dim_pca,
                neighbors,
                min_dist,
                components,
                cluster_size,
                min_sample,
                epsilon,
            ) = idx

            key_list = []
            noise_len_list = []
            num_of_classes_list = []
            for idx, row in group.iterrows():
                key = int(row["key"])
                data_path = Path(
                    parent_dir,
                    exp_name,
                    f"t1_r{corr}_npca{dim_pca}/neighbours{neighbors}_"
                    f"mindist{min_dist}_dim{components}/dat_nc/cluster_out_{key}.nc",
                )
                xda = xr.load_dataarray(data_path).stack(z=["x", "y"]).dropna(dim="z")
                total_len = xda.shape[0]
                noise_len = xda.where(xda == -1, drop=True).shape[0]
                noise_len_rel = round(noise_len / total_len * 100, 2)
                key_list.append(key)
                noise_len_list.append(noise_len_rel)
                num_of_classes_list.append(int(row["num_of_classes"]))

            new_results = (
                group.drop(["key", "num_of_classes"], axis=1)
                .groupby(group_cols)
                .sum()
                .reset_index()
            )
            new_results["key"] = [key_list for _ in new_results.index]
            new_results["num_of_classes"] = [
                num_of_classes_list for _ in new_results.index
            ]
            new_results["noise_avg"] = round(np.mean(noise_len_list), 2)
            new_results["noise_rel"] = [noise_len_list for _ in new_results.index]

            new_results_list.append(new_results)

        new_results_df = pd.concat(new_results_list)
        print("Grouped results sum per umap hyperparam:")
        print(new_results_df.groupby("umap_n_neighbors").sum()["result"])
        print(new_results_df.groupby("umap_min_dist").sum()["result"])
        new_results_path = Path(
            str(results_path).replace("results.csv", "results_good.csv")
        )
        new_results_df.to_csv(new_results_path)
        new_results_df.query("result > 3").to_csv(
            str(new_results_path).replace("good.csv", "good_filtered.csv")
        )

    if (eval_type == 2) or (eval_type == 3):
        ## generate heatmaps that show num_of_classes
        # loop over some metrics to categorize analysis
        for (
            corr,
            dim_pca,
        ) in product(df_results["corr"].unique(), df_results["dim_pca"].unique()):
            print(f"..calc {corr = } | {dim_pca = }..")
            group_calc, groups_drop = get_groups(experiment=exp_name)
            # verify that gropped groups contain no multiple values
            dropped_params = []
            for group in groups_drop:
                assert df_results[group].unique().shape[0] == 1
                dropped_params.append(f"{group} = {df_results[group].unique()[0]}")
            title_str_sub = " | ".join(dropped_params)

            df_iter = df_results.query(f"corr == {corr}").query(f"dim_pca == {dim_pca}")
            if df_iter.shape[0] < 1:
                # skip this combination of corr/dim_pca if empty
                continue

            groups_drop = groups_drop + ["time_period", "corr", "dim_pca"]
            df_iter = df_iter.drop(groups_drop, axis=1)

            concat_iter, group_n = group_df_min_median_max(
                df_in=df_iter, groups=group_calc
            )
            title_str = f"{corr = } | {dim_pca = } | {group_n = }"

            fig, axes = pplt.subplots()
            axes.heatmap(
                concat_iter,
                vmin=1,
                vmax=13,
                lw=1,
                labels=True,
                ec="k",
                clip_on=False,
                cmap="coolwarm",
                colorbar="t",
            )
            axes.format(title=f"{title_str}\n{title_str_sub}")

            save_path = Path(
                results_path.parent, f"hm_r{corr}_d{dim_pca}{str_suffix}.png"
            )
            fig.save(save_path)
