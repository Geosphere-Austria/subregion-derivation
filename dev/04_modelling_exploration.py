"""modelling of subregions with various clustering approaches"""
from pathlib import Path

from utils import (
    load_climate_normals_and_reshape,
    create_empty_dataset_for_clusters,
    calc_kmeans_clusters,
    calc_spectral_clustering,
    calc_dbscan_clusters,
    calc_tsne,
    add_cluster_result_to_datasets,
    plot_cluster,
)

if __name__ == "__main__":
    variables_file = Path(
        "dat/interim/02_preprocessed_climate_normals/indicators_climate_normals.nc"
    )
    da_data = load_climate_normals_and_reshape(file_path=variables_file)

    # clean data
    da_clean = da_data.dropna(dim="z")

    # clustering methods
    methods = [
        "kmeans",
        # "tsne",
        # "dbscan",
        "spectral_clustering",
    ]

    ds_cluster = create_empty_dataset_for_clusters(
        da_original=da_data, methods_list=methods
    )

    # modelling
    ## config params
    N_CLUSTERS = 5

    ## kmeans
    if "kmeans" in methods:
        print("..calculate kmeans..")
        ds_kmeans = calc_kmeans_clusters(data=da_clean, n_clusters=N_CLUSTERS)
        ds_cluster = add_cluster_result_to_datasets(ds_in=ds_cluster, cluster=ds_kmeans)
        plot_cluster(data=ds_cluster["kmeans"], save=True)

    ## spectral clustering
    if "spectral_clustering" in methods:
        print("..calculate spectral clustering..")
        ds_spec_clust = calc_spectral_clustering(data=da_clean, n_clusters=5)
        ds_cluster = add_cluster_result_to_datasets(
            ds_in=ds_cluster, cluster=ds_spec_clust
        )
        plot_cluster(data=ds_cluster["spectral_clustering"], save=True)

    ## dbscan
    if "dbscan" in methods:
        print("..calculate dbscan..")
        ds_dbscan = calc_dbscan_clusters(data=da_clean, eps=0.5, min_samples=5)
        ds_cluster = add_cluster_result_to_datasets(ds_in=ds_cluster, cluster=ds_dbscan)
        plot_cluster(data=ds_cluster["dbscan"], save=True)

    ## TSNE
    if "tsne" in methods:
        print("..calculate tsne..")
        ds_tsne = calc_tsne(data=da_clean, perplexity=100, n_iter=500)
        ds_cluster = add_cluster_result_to_datasets(ds_in=ds_cluster, cluster=ds_tsne)
        plot_cluster(data=ds_cluster["tsne"], save=True)

    # save data
    ds_cluster.to_netcdf(Path("dat/interim/clusters_dataset.nc"))
