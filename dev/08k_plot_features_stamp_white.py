"""create a stampplot for input features"""

from matplotlib import ticker
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray
import numpy as np
import seaborn as sns


def load_input_features():
    xda_ = xr.open_dataarray("dat/interim/07_cluster_evaluation/features.nc")
    xda_ = xda_.rio.write_crs(3416)
    xda_ = xda_.drop(
        [
            "spatial_ref",
        ]
    )
    return xda_.transpose("variable", "y", "x")


features = load_input_features()
clusters_tif = "dat/out/physioclimatic_clusters_raster_AT.tif"
clusters = rioxarray.open_rasterio(clusters_tif).squeeze()
clusters = clusters.where(clusters != -9999, np.nan)


def stampplot_austria(data):
    dim_len = len(data["variable"])
    print(f"number of features: {dim_len}")
    fig, axes = plt.subplots(
        ncols=6,
        nrows=5,
        figsize=(20, 12),
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    fg_color = "k"
    bg_color = "w"
    axes = axes.ravel()
    axit = iter(axes)
    for var_name in data["variable"]:
        feature = data.sel(variable=var_name)
        ax = next(axit)
        feature.isnull().plot(ax=ax, add_colorbar=False, color=bg_color)
        # feature.plot(ax=ax, cbar_kwargs={"orientation": "horizontal", "label": str(var_name.values)})
        vmin = feature.quantile(0.01).values
        vmax = feature.quantile(0.99).values
        cmap = "viridis"
        if vmin < 0 and vmax > 0:
            vabsmax = max(abs(vmin), abs(vmax))
            vmin = -vabsmax
            vmax = vabsmax
            hue_neg, hue_pos = 250, 15
            cmap = sns.diverging_palette(
                hue_neg, hue_pos, s=100, l=60, center="dark", as_cmap=True
            )

        p = feature.plot(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_kwargs={
                "orientation": "horizontal",
                "label": "",
                "location": "bottom",
            },
        )

        # ax.text(.013, .975, str(var_name.values), ha="left", va="top", transform=ax.transAxes, color="w", backgroundcolor="k", fontsize=8)
        p.colorbar.ax.xaxis.set_ticks_position("top")
        p.colorbar.ax.xaxis.set_label_position("top")
        ax.annotate(
            str(var_name.values),
            (0, 1),
            xytext=(4, -4),
            xycoords="axes fraction",
            textcoords="offset points",
            color=fg_color,
            bbox=dict(edgecolor=fg_color, facecolor=bg_color),
            ha="left",
            va="top",
            fontsize=7,
        )
        ax.set_aspect("equal")
        ax.set_title("")

        cb = ax.collections[-1].colorbar
        cb.ax.xaxis.set_tick_params(color=fg_color)
        cb.outline.set_edgecolor(fg_color)
        plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=fg_color)
        # set number of xticks in colorbar
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.ax.xaxis.set_major_locator(ticker.AutoLocator())
        cb.locator = tick_locator
        cb.update_ticks()

    for ax in axes[dim_len:]:
        ax.set_visible(False)
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_color(fg_color)
        ax.spines["top"].set_color(fg_color)
        ax.spines["right"].set_color(fg_color)
        ax.spines["left"].set_color(fg_color)
        ax.tick_params(axis="x", colors=fg_color, which="both")
        ax.tick_params(axis="y", colors=fg_color, which="both")
        ax.yaxis.label.set_color(fg_color)
        ax.xaxis.label.set_color(fg_color)
    fig.set_facecolor(bg_color)
    return None


stampplot_austria(data=features)
plt.savefig("plt/fig_stamp_features_white.png", dpi=300)
