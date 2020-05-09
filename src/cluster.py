"""
This module contains functions for visualizing data and model results

FUNCTIONS

    silplot()
        Generates silhouette subplot of kmeans clusters alongside PCA n=2

    display_gapstat_with_errbars()
        Generates plots of gap stats with error bars for each number of
        clusters

    fit_neighbors()
        Fits n nearest neighbors based on min samples and returns distances

    plot_epsilon()
        Plot epsilon by index sorted by increasing distance

    silscore_dbscan()
        Generates sil score ommitting observations not assigned to any cluster
        by dbscan

    fit_dbscan():
        Fits dbscan and returns dictionary of results including model, labels,
        indices
    
    print_dbscan_results():
        Prints summary results of fitted dbscan_dict

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from gap_statistic import OptimalK

from .visualize import plot_line, plot_value_counts

# Define plotting function to generate plot of gap stats with error bars


def silplot(X, cluster_labels, clusterer, pointlabels=None):
    """Generates silhouette subplot of kmeans clusters alongside PCA n=2

    Source: The majority of the code from this function was provided as a
            helper function from the CS109b staff in HW2

            The original code authored by the cs109b teaching staff
            is modified from:
            http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
 
    """
    
    n_clusters = clusterer.n_clusters
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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
    for i in range(0,n_clusters+1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.grid(':', alpha=0.5)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X) 
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], marker='.', s=200, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    xs = X_pca[:, 0]
    ys = X_pca[:, 1]    

    
    if pointlabels is not None:
        for i in range(len(xs)):
            plt.text(xs[i],ys[i],pointlabels[i])

    # Labeling the clusters (transform to PCA space for plotting)
    centers = pca.transform(clusterer.cluster_centers_)
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % int(i), alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("PCA-based visualization of the clustered data.")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(':', alpha=0.5)

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data "\
        "with n_clusters = {},\naverage silhouette score: {:.4f}"\
        "".format(n_clusters, silhouette_avg),
        fontsize=14,
        fontweight='bold',
        y=1.08
    )
    
    plt.tight_layout()
    plt.show()


def display_gapstat_with_errbars(gap_df, height=4):
    """Generates plots of gap stats with error bars for each number of clusters
    """
    gaps = gap_df["gap_value"].values
    diffs = gap_df["diff"]
    
    err_bars = np.zeros(len(gap_df))
    err_bars[1:] = diffs[:-1] - gaps[:-1] + gaps[1:]

    fig, ax = plt.subplots(figsize=(12,height))

    plt.title('Gap statistic with error bars by number of clusters', fontsize=19)

    plt.scatter(gap_df["n_clusters"], gap_df["gap_value"], color='k')
    plt.errorbar(
        gap_df["n_clusters"],
        gap_df["gap_value"],
        yerr=err_bars,
        capsize=6,
        color='k'
    )
    
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Gap Statistic", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.grid(':', alpha=0.4)
    plt.tight_layout()
    plt.show()
    

# Define functions for identifying epsilon values, fitting dbscan, and evaluating results

def fit_neighbors(data, min_samples):
    """Fits n nearest neighbors based on min samples and returns distances
    """
    fitted_neigbors = NearestNeighbors(n_neighbors=min_samples).fit(data)
    distances, indices = fitted_neigbors.kneighbors(data)
    return distances


def plot_epsilon(distances, min_samples):
    """Plot epsilon by index sorted by increasing distance
    """
    fig, ax = plt.subplots(figsize=(12,7))

    plt.title(
        '{}-NN distance (epsilon) by sorted index'.format(min_samples-1),
        fontsize=19
    )
    
    dist_to_nth_nearest_neighbor = distances[:,-1]
    plt.plot(np.sort(dist_to_nth_nearest_neighbor), color='k')
    plt.xlabel("Index (sorted by increasing distances)", fontsize=16)
    plt.ylabel("$\epsilon$", fontsize=18)
    plt.tick_params(right=True, labelright=True, labelsize=14)
    plt.grid(':', alpha=0.4)
    plt.tight_layout()
    plt.show()

    
def silscore_dbscan(data, labels, clustered_bool):
    """Generates sil score ommitting observations not assigned to any cluster by dbscan 
    """
    return silhouette_score(data[clustered_bool], labels[clustered_bool])


def fit_dbscan(data, min_samples, eps):
    """Fits dbscan and returns dictionary of results including model, labels, indices
    """
    fitted_dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    db_labels = fitted_dbscan.labels_
    n_clusters = sum([i != -1 for i in set(db_labels)])
    
    # generate boolean indices for observations assigned to clusters
    clustered_bool = [i != -1 for i in db_labels]
    
    dbscan_dict = {
        'model': fitted_dbscan,
        'n_clusters': n_clusters,
        'labels': db_labels,
        'core_sample_indices': fitted_dbscan.core_sample_indices_,
        'clustered_bool': clustered_bool,
        'cluster_counts': pd.Series(db_labels).value_counts(),
        'sil_score': silscore_dbscan(data, db_labels, clustered_bool)
                     if n_clusters>1 else 0
    }
    return dbscan_dict


def print_dbscan_results(dbscan_dict):
    """Prints summary results of fitted dbscan_dict
    """
    eps = dbscan_dict['model'].eps
    min_samples = dbscan_dict['model'].min_samples
    n_samples = len(dbscan_dict['labels'])
    n_unclustered = dbscan_dict['cluster_counts'].loc[-1]
    n_clusters = dbscan_dict['n_clusters']
    
    # print basic summary info
    print(
        '\nFor the DBSCAN model:\n\n{}\n\n'\
        '{} {} identified, and {:,} of the n={:,} observations '\
        'were not assigned to any clusters.\n\n'\
        'The distribution of resulting labels are illustrated by this chart '\
        'with un-clustered observations represented by the cluster labeled -1.\n'\
        ''.format(
            dbscan_dict['model'],
            n_clusters,
            'clusters were' if n_clusters>1 else 'cluster was',
            n_unclustered,
            n_samples
        )
    )
    
    # plot distribution of labels
    plot_value_counts(dbscan_dict['cluster_counts'], 'DBSCAN cluster')
    
    # print silhouette score
    if n_clusters>1:
        print(
            '\nThe resulting silhouette score, excluding the points not '\
            'assigned to any cluster is:\n\n\t{:.4f}\n'.format(
                dbscan_dict['sil_score']
            )
        )
    
    else:
        print(
            '\nBecause only 1 cluster has been assigned by the algorithm, the '\
            'resulting silhouette score of 0 has been assigned to these results.\n'
        )