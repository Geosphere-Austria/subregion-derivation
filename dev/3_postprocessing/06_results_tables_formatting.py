"""This script loads and formats tables containing the hyperparameter tuning results"""

import pandas as pd

results = pd.read_csv("dat/interim/05_hyperparametertuning/experiment_hdbscan_hyperparams/results.csv", index_col=0)
results = results.drop(["time_period", "corr", "metric"], axis=1)
results.to_csv("dat/out/hyperparametertuning_results.csv")
