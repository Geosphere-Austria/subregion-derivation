"""small script to analyse hdbscan good results table"""
from tabulate import tabulate
import pandas as pd

df = pd.read_csv(
    "dat/interim/05_hyperparametertuning/experiment_hdbscan_hyperparams/results_good_filtered.csv",
    index_col=0,
)

df.columns = df.columns.str.replace("umap_n_neighbors", "u neighb")
df.columns = df.columns.str.replace("umap_min_dist", "u delta")
df.columns = df.columns.str.replace("hs_min_cluster_size", "hs cluster size")
df.columns = df.columns.str.replace("hs_min_sample", "hs samples")
df.columns = df.columns.str.replace("hs_cluster_selection_epsilon", "hs epsilon")

df_short = df.drop(["corr", "dim_pca", "umap_n_components"], axis=1)
df_agg = df_short.drop(["key", "noise_rel"], axis=1)

print(df_short)

print(df_agg)

print(
    "Look at aggregates result numbers per parameter value (groupby based on variable in first column)"
)
varlist = ["u neighb", "u delta", "hs cluster size", "hs samples", "hs epsilon"]
for var in varlist:
    df_iter = df_agg.groupby(var).mean()
    print(f"\n#### Aggregated mean groupby {var}\n")
    print(tabulate(df_iter, headers="keys", tablefmt="github"))

df_shorter = df_agg.query("noise_avg < 8").reset_index().drop("index", axis=1)
print("Aggregated over all parameters")
print(tabulate(df_shorter, headers="keys", tablefmt="github"))
