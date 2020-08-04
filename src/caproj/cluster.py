"""
caproj.cluster
~~~~~~~~~~~~~~

This module contains functions for visualizing data and model results

**Module classes:**

.. autosummary::

   UMAP_embedder

**Module functions:**

.. autosummary::

   silplot
   display_gapstat_with_errbars
   fit_neighbors
   plot_epsilon
   silscore_dbscan
   fit_dbscan
   print_dbscan_results
   plot_dendrogram
   plot_cluster_hist
   plot_umap_scatter
   plot_category_scatter

"""

from math import pi

import hdbscan
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score

from .visualize import plot_value_counts, save_plot


def silplot(
    X, cluster_labels, clusterer, pointlabels=None, height=6, savepath=None
):
    """Generates silhouette subplot of kmeans clusters alongside PCA n=2

    Two side-by-side subplots are generated showing (1) the silhouette plot of
    the clusterer's results and (2) the PCA 2-dimensional reduction of the
    input data, color-coded by cluster.

    Source: The majority of the code from this function was provided as a
            helper function from the CS109b staff in HW2

            The original code authored by the cs109b teaching staff
            is modified from:
            http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    :param X: original mutli-dimensional data values against which you are plotting
    :type X: pandas.DataFrame
    :param cluster_labels: list of labels for each observation in ``X``
    :type cluster_labels: list or array
    :param clusterer: fitted sklearn kmeans clustering object
    :type clusterer: sklearn.cluster.KMeans object
    :param pointlabels: list of labels for each point, defaults to None
    :type pointlabels: list or None, optional
    :param height: height of resulting subplots, defaults to 6
    :type height: int, optional
    :param savepath: string or None, filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    """
    n_clusters = clusterer.n_clusters

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, height))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(0, n_clusters + 1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels == i
        ]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters", fontsize=14)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=12)
    ax1.set_ylabel("Cluster label", fontsize=12)
    ax1.grid(":", alpha=0.5)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    ax2.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        marker=".",
        s=200,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )
    xs = X_pca[:, 0]
    ys = X_pca[:, 1]

    if pointlabels is not None:
        for i in range(len(xs)):
            plt.text(xs[i], ys[i], pointlabels[i])

    # Labeling the clusters (transform to PCA space for plotting)
    centers = pca.transform(clusterer.cluster_centers_)
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(
            c[0], c[1], marker="$%d$" % int(i), alpha=1, s=50, edgecolor="k"
        )

    ax2.set_title("PCA-based visualization of the clustered data", fontsize=14)
    ax2.set_xlabel("principal component 1", fontsize=12)
    ax2.set_ylabel("principal component 2", fontsize=12)
    ax2.grid(":", alpha=0.5)

    plt.suptitle(
        "Silhouette analysis, K-means clustering on sample data "
        "with n_clusters = {},\naverage silhouette score: {:.4f}"
        "".format(n_clusters, silhouette_avg),
        fontsize=18,
        y=1,
    )

    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def display_gapstat_with_errbars(gap_df, height=4, savepath=None):
    """Generates plots of gap stats with error bars for each number of clusters

    :param gap_df: dataframe attribute of a fitted ``gap_statistic.OptimalK`` object
            for plotting (i.e. ``OptimalK.gap_df``)
    :type gap_df: DataFrame
    :param height: hieght of the resulting plot, defaults to 4
    :type height: int, optional
    :param savepath: filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    :type savepath: str or None
    """
    gaps = gap_df["gap_value"].values
    diffs = gap_df["diff"]

    err_bars = np.zeros(len(gap_df))
    err_bars[1:] = diffs[:-1] - gaps[:-1] + gaps[1:]

    fig, ax = plt.subplots(figsize=(12, height))

    plt.title(
        "Gap statistic with error bars by number of clusters", fontsize=19
    )

    plt.scatter(gap_df["n_clusters"], gap_df["gap_value"], color="k")
    plt.errorbar(
        gap_df["n_clusters"],
        gap_df["gap_value"],
        yerr=err_bars,
        capsize=6,
        color="k",
    )

    plt.xlabel(r"number of clusters $K$", fontsize=16)
    plt.ylabel("gap statistic", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.grid(":", alpha=0.4)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


# Define functions for identifying epsilon values, fitting dbscan,
# and evaluating results


def fit_neighbors(data, min_samples):
    """Fits n nearest neighbors based on min samples and returns distances

    This is a simple implementation of the ``sklearn.neighbors.NearestNeighbors``
    and returns the distance results from that object's ``fitted_neighbors`` method

    :param data: data on which to perform nearest neighbors algorithm
    :type data: dataframe or array
    :param min_samples: number of neighbors to use by default for kneighbors queries
    :type min_samples: int
    :return: array representing the lengths to points
    :rtype: array
    """
    fitted_neigbors = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, indices = fitted_neigbors.kneighbors(data)
    return distances


def plot_epsilon(distances, min_samples, height=5, savepath=None):
    """Plot epsilon by index sorted by increasing distance

    Generates a line plot of epsilon with observations sorted by increasing distances

    :param distances: distances generated by :func:`fit_neighbors`
    :type distances: array
    :param min_samples: number of neighbors used to generate distances
    :type min_samples: int
    :param height: height of plot, defaults to 5
    :type height: int, optional
    :param savepath: filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    :type savepath: str or None
    """
    fig, ax = plt.subplots(figsize=(12, height))

    plt.title(
        "{}-NN distance (epsilon) by sorted index".format(min_samples - 1),
        fontsize=19,
    )

    dist_to_nth_nearest_neighbor = distances[:, -1]
    plt.plot(np.sort(dist_to_nth_nearest_neighbor), color="k")
    plt.xlabel("Index (sorted by increasing distances)", fontsize=16)
    plt.ylabel(r"$\epsilon$", fontsize=18)
    plt.tick_params(right=True, labelright=True, labelsize=14)
    plt.grid(":", alpha=0.4)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def silscore_dbscan(data, labels, clustered_bool):
    """Generates sil score ommitting observations not assigned to any cluster by dbscan

    :param data: original data used for dbscan clustering
    :type data: array or dataframe
    :param labels: cluster label for each observation in ``data``
    :type labels: array
    :param clustered_bool: boolean value for each observation indicated whether it had
            been clustered by dbscan
    :type clustered_bool: list or 1-d array
    :return: silhouette score
    :rtype: float
    """
    return silhouette_score(data[clustered_bool], labels[clustered_bool])


def fit_dbscan(data, min_samples, eps):
    """Fits dbscan and returns dictionary of results including model, labels, indices

    :param data: original data to be fitted using ``sklearn.cluster.DBSCAN``
    :type data: array-like
    :param min_samples: The number of samples (or total weight) in a neighborhood for
            a point to be considered as a core point. This includes the point itself.
    :type min_samples: int
    :param eps: The maximum distance between two samples for one to be considered as
            in the neighborhood of the other. This is not a maximum bound on the
            distances of points within a cluster. This is the most important DBSCAN
            parameter to choose appropriately for your data set and distance function.
    :type eps: int or float
    :return: dictionary of results and important characteristics of the fitted
            DBSCAN algorithm (see NOTE below)
    :rtype: dict

    NOTE: Dictionary returned includes the following items::

       {
           "model": DBSCAN(eps=eps, min_samples=min_samples).fit(data),
           "n_clusters": sum([i != -1 for i in set(model.labels_)]),
           "labels": model.labels_,
           "core_sample_indices": model.core_sample_indices_,
           "clustered_bool": [i != -1 for i in labels],
           "cluster_counts": pd.Series(labels).value_counts(),
           "sil_score": silscore_dbscan(data, labels, clustered_bool),
       }

    """
    fitted_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    db_labels = fitted_dbscan.labels_
    n_clusters = sum([i != -1 for i in set(db_labels)])

    # generate boolean indices for observations assigned to clusters
    clustered_bool = [i != -1 for i in db_labels]

    dbscan_dict = {
        "model": fitted_dbscan,
        "n_clusters": n_clusters,
        "labels": db_labels,
        "core_sample_indices": fitted_dbscan.core_sample_indices_,
        "clustered_bool": clustered_bool,
        "cluster_counts": pd.Series(db_labels).value_counts(),
        "sil_score": silscore_dbscan(data, db_labels, clustered_bool)
        if n_clusters > 1
        else 0,
    }
    return dbscan_dict


def print_dbscan_results(dbscan_dict, savepath=None):
    """Prints summary results of fitted DBSCAN results dictionary

    Provides printed summary and plotted value counts by cluster

    :param dbscan_dict: returned output dictionary from :func:`fit_dbscan`` function
    :type dbscan_dict: dict
    :param savepath: filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    :type savepath: str or None
    """
    eps = dbscan_dict["model"].eps
    min_samples = dbscan_dict["model"].min_samples
    n_samples = len(dbscan_dict["labels"])
    n_unclustered = dbscan_dict["cluster_counts"].loc[-1]
    n_clusters = dbscan_dict["n_clusters"]

    # print basic summary info
    print(
        "\nFor the DBSCAN model:\n\n{}\n\n"
        "{} {} identified, and {:,} of the n={:,} observations "
        "were not assigned to any clusters.\n\n"
        "The distribution of resulting labels are illustrated by this chart "
        "with un-clustered observations represented by the cluster labeled -1.\n"
        "".format(
            dbscan_dict["model"],
            n_clusters,
            "clusters were" if n_clusters > 1 else "cluster was",
            n_unclustered,
            n_samples,
        )
    )

    # plot distribution of labels
    plot_value_counts(dbscan_dict["cluster_counts"])
    plt.title(
        r"DBSCAN clustering results, min samples$={}$ and $\epsilon={}$"
        r"".format(min_samples, eps),
        fontsize=16,
    )
    plt.xlabel("DBSCAN clusters (-1 indicates unclustered)", fontsize=12)
    plt.ylabel("number of observations", fontsize=12)
    plt.tight_layout()
    save_plot(plt_object=plt, savepath=savepath)
    plt.show()

    # print silhouette score
    if n_clusters > 1:
        print(
            "\nThe resulting silhouette score, excluding the points not "
            "assigned to any cluster is:\n\n\t{:.4f}\n".format(
                dbscan_dict["sil_score"]
            )
        )

    else:
        print(
            "\nBecause only 1 cluster has been assigned by the algorithm, the "
            "resulting silhouette score of 0 has been assigned to these results.\n"
        )


def plot_dendrogram(
    linkage_data,
    method_name,
    yticks=16,
    ytick_interval=1,
    height=4.5,
    savepath=None,
):
    """Plots a dendrogram given a set of input hierarchy linkage data

    :param linkage_data: np.array output from scipy.cluster.hierarchy, which
                         should have been applied to a distance matrix to
                         convert it to linkage data
    :param method_name: string describing the linkage method used, should
                        be fewer than 30 characters
    :param yticks: integer, the number of desired y tick lavels for the
                   resulting plot
    :param ytick_interval: integer, the desired interval for the resulting
                           y ticks
    :param height: float, the desired height of the resulting plot
    :param savepath: string or None, filepath at which to save generated plot,
                     if None, no file will be saved, defaults to None
    """

    plt.figure(figsize=(12, height))

    hac.dendrogram(
        linkage_data, above_threshold_color="lightgray", orientation="top"
    )

    plt.title(
        "Agglomerative clustering dendrogram (using {})".format(method_name),
        fontsize=18,
    )
    plt.grid(":", axis="y", alpha=0.5)
    plt.yticks(np.arange(0, yticks + 1, ytick_interval), fontsize=10)
    plt.ylabel("distance", fontsize=14)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


# from dataclasses import dataclass


# from pickle import dump, load
# import plotly.io as pio
# import plotly.express as px
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from IPython.display import Image, SVG
# init_notebook_mode()
# pio.renderers.keys()
# pio.renderers.default = 'jupyterlab'


class UMAP_embedder:
    """Class methods for generating UMAP embedding and HDBSCAN clusters
    """

    def __init__(
        self, scaler, final_cols, mapper_dict, clusterer, bert_embedding
    ):
        # self.initial_columns = columns
        self.initial_columns = [
            "PID",
            "Project_Name",
            "Description",
            "Category",
            "Borough",
            "Managing_Agency",
            "Client_Agency",
            "Phase_Start",
            "Current_Project_Years",
            "Current_Project_Year",
            "Design_Start",
            "Budget_Start",
            "Schedule_Start",
            "Final_Change_Date",
            "Final_Change_Years",
            "Phase_End",
            "Budget_End",
            "Schedule_End",
            "Number_Changes",
            "Duration_Start",
            "Duration_End",
            "Schedule_Change",
            "Budget_Change",
            "Schedule_Change_Ratio",
            "Budget_Change_Ratio",
            "Budget_Abs_Per_Error",
            "Budget_Rel_Per_Error",
            "Duration_End_Ratio",
            "Budget_End_Ratio",
            "Duration_Ratio_Inv",
            "Budget_Ratio_Inv",
        ]
        # self.scale_cols = df_to_transform.columns
        self.scale_cols = [
            "Current_Project_Years",
            "Current_Project_Year",
            "Budget_Start",
            "Final_Change_Years",
            "Budget_End",
            "Number_Changes",
            "Duration_Start",
            "Duration_End",
            "Schedule_Change",
            "Budget_Change",
            "Schedule_Change_Ratio",
            "Budget_Change_Ratio",
            "Budget_Abs_Per_Error",
            "Budget_Rel_Per_Error",
            "Duration_End_Ratio",
            "Budget_End_Ratio",
            "Duration_Ratio_Inv",
            "Budget_Ratio_Inv",
        ]
        self.scaler = scaler
        self.cols_to_dummify = [
            "Borough",
            "Category",
            "Client_Agency",
            "Managing_Agency",
            "Phase_Start",
            "Budget_Start",
            "Duration_Start",
        ]
        # self.cols_to_dummify = columns_before_dummified
        self.final_cols = final_cols
        self.mapper_dict = mapper_dict
        self.clusterer = clusterer
        self.embedding = bert_embedding

    def get_mapping_attributes(self, df, return_extra=False, dimensions="all"):
        """
        if return extra = True, returns 3 objects:
            0. mapping
            1. columns needed to be added to harmonize with entire data
            2. dummified df before adding columns of [1]
        """
        df_to_transform = df[self.scale_cols]  # .drop(columns=["PID"])
        transformed_columns = pd.DataFrame(
            self.scaler.transform(df_to_transform),
            columns=df_to_transform.columns,
        )
        scaled_df = (
            df[df.columns.difference(transformed_columns.columns)]
        ).join(transformed_columns)
        dummified = pd.get_dummies(scaled_df[self.cols_to_dummify])

        added_cols = set(self.final_cols) - set(dummified.columns)
        added_cols = {col: 0 for col in added_cols}

        dummified_full = dummified.assign(**added_cols)
        dummified_full = dummified_full[self.final_cols]
        mapping_df_list = []
        mapper_list = (
            self.mapper_dict["attributes"].values()
            if dimensions == "all"
            else [
                self.mapper_dict["attributes"][dimension]
                for dimension in dimensions
            ]
        )

        for mapper in mapper_list:
            mapping = mapper.transform(dummified_full)
            mapping_df = pd.DataFrame(
                mapping,
                columns=[
                    f"umap_attributes_{mapping.shape[1]}D_embed_{col+1}"
                    for col in range(mapping.shape[1])
                ],
            )
            mapping_df_list.append(mapping_df)

        final_df = pd.concat(mapping_df_list, axis=1)
        final_df["PID"] = scaled_df["PID"]

        if return_extra:
            return final_df, added_cols, scaled_df, dummified
        else:
            return final_df

    def get_mapping_description(self, df, dimensions="all"):
        """Returns UMAP final dataframe
        """
        merged = (
            df[["PID"]]
            .merge(self.embedding, on="PID", how="left")
            .drop(columns="PID")
        )
        mapping_df_list = [merged]
        # mapping_columns = [list(self.embedding.columns.copy())]
        mapper_list = (
            self.mapper_dict["description"].values()
            if dimensions == "all"
            else [
                self.mapper_dict["description"][dimension]
                for dimension in dimensions
            ]
        )
        for mapper in mapper_list:
            mapping = mapper.transform(merged)
            mapping_df = pd.DataFrame(
                mapping,
                columns=[
                    f"umap_descr_{mapping.shape[1]}D_embed_{col+1}"
                    for col in range(mapping.shape[1])
                ],
            )
            mapping_df_list.append(mapping_df)
        # mapping_columns += list(mapping_df.columns.copy())

        final_df = pd.concat(mapping_df_list, axis=1)
        final_df["PID"] = df["PID"].values

        return final_df

    def get_full_df(self, df, dimensions="all"):
        """Returns UMAP full dataframe
        """
        attribute_df = self.get_mapping_attributes(df, dimension="all")
        description_df = self.get_mapping_description(df)
        labels, probabilities = self.get_clustering(
            attribute_df[
                ["umap_attributes_2D_embed_1", "umap_attributes_2D_embed_2"]
            ]
        )
        full_df = description_df.merge(attribute_df, on="PID", how="left")
        full_df["PID"] = attribute_df["PID"].values
        full_df["attribute_clustering_label"] = labels
        return full_df

    def get_clustering(self, attributes_2D_mapping):
        """Returns HDBSCAN cluster labels
        """
        assert attributes_2D_mapping.shape[1] == 2
        new_labels = hdbscan.approximate_predict(
            self.clusterer, attributes_2D_mapping
        )
        return new_labels


def make_spider(mean_peaks_per_cluster, row, name, color):
    """Generate spider plot showing attributes of a single cluster
    """
    # number of variable
    categories = list(mean_peaks_per_cluster)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot?
    # (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(3, 2, row + 1, polar=True,)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color="grey", size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    # plt.ylim(0,40)

    values = (
        mean_peaks_per_cluster.loc[row].drop("group").values.flatten().tolist()
    )
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(name, size=14, color=color, y=1.1)


def plot_spider_clusters(title, mean_peaks_per_cluster, savepath=None):
    """Generate spider plot subplots for all input clusters
    """
    my_dpi = 50

    fig = plt.figure(figsize=(600 / my_dpi, 700 / my_dpi), dpi=my_dpi + 40)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(mean_peaks_per_cluster.index))

    # Loop to plot
    for row in range(0, len(mean_peaks_per_cluster.index)):
        make_spider(
            mean_peaks_per_cluster,
            row=row,
            name="cluster "
            + mean_peaks_per_cluster["group"][row].astype("str"),
            color=my_palette(row),
        )
    fig.suptitle(title, fontsize=18, y=1)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def plot_cluster_hist(
    data,
    title,
    metric,
    cluster_col="cluster",
    val_col="Standardized Metric Value",
    metric_col="Metric",
    cmap="Paired",
    bins=6,
    savepath=None,
):
    """Requires melted dataframe as input and plots histograms by cluster
    """
    cluster_list = sorted(list(set(data[cluster_col].values)))

    fig, ax = plt.subplots(figsize=(12, 4))

    plt.title(title, fontsize=18)

    for clus in cluster_list:
        plt.hist(
            data.loc[
                (data[metric_col] == metric) & (data[cluster_col] == clus)
            ][val_col],
            bins=bins,
            alpha=0.5,
            label=clus,
        )

    plt.xlabel(val_col, fontsize=14)
    plt.ylabel("number of observations", fontsize=14)
    plt.legend(edgecolor="k", title=cluster_col, fontsize=12)
    plt.grid(":", alpha=0.5)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def plot_umap_scatter(
    x,
    y,
    color,
    title,
    scale_var,
    colormap="Reds",
    xlabel="1st dimension",
    ylabel="2nd dimension",
    savepath=None,
):
    """plots scatterplot with color scale
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    plt.title(title, fontsize=18)

    scatter = plt.scatter(
        x, y, c=color, alpha=0.5, s=120, edgecolor="k", cmap=colormap
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.colorbar(scatter, label=scale_var.replace("_", " "))
    ax.grid(":", alpha=0.5)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()


def plot_category_scatter(
    data,
    x_col,
    y_col,
    cat_col,
    title,
    colormap="Paired",
    xlabel="1st dimension",
    ylabel="2nd dimension",
    savepath=None,
):
    """plots scatterplot with categories colors
    """
    cat_list = sorted(list(set(data[cat_col].values)))

    fig, ax = plt.subplots(figsize=(12, 4))

    plt.title(title, fontsize=18)

    for cat in cat_list:
        plt.scatter(
            data.loc[data[cat_col] == cat][x_col],
            data.loc[data[cat_col] == cat][y_col],
            alpha=0.5,
            s=120,
            edgecolor="k",
            label=cat,
        )
    plt.legend(
        fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
    )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.grid(":", alpha=0.5)
    plt.tight_layout()

    save_plot(plt_object=plt, savepath=savepath)

    plt.show()
