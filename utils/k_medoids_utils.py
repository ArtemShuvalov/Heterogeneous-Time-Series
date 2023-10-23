import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

import seaborn as sns
import matplotlib.pyplot as plt


def get_coocurrence_matrix(dtw_matrix: np.array,
                           n_runs: int = 1000,
                           clusters_possible_range: range = range(5, 30)) -> tuple:
    """
    Compute the co-occurrence matrix and perform consensus clustering.

    Args:
    - dtw_matrix (np.array): A precomputed distance matrix.
    - n_runs (int): Number of runs for KMedoids clustering. Default is 1000.
    - clusters_possible_range (range): Range of possible cluster sizes. Default is range(5, 30).

    Returns:
    tuple: A tuple containing the following elements:
    - clusters_possible_range (range): The range of possible cluster sizes.
    - silhouette_scores (list): Silhouette scores for each cluster size in clusters_possible_range.
    - consensus_labels (array): Consensus labels obtained from consensus clustering.

    This function calculates the co-occurrence matrix by running KMedoids clustering multiple times
    and then performs consensus clustering to obtain consensus labels. Silhouette scores are
    calculated for each cluster size in the specified range to evaluate the quality of clustering.

    Example:
    clusters_range, silhouette_scores, consensus_labels = get_cooccurrence_matrix(dtw_matrix, n_runs=1000)
    """

    n_samples = dtw_matrix.shape[0]

    silhouette_scores = []

    for num_clusters in clusters_possible_range:
        # Initialize co-occurrence matrix
        co_occurrence_matrix = np.zeros((n_samples, n_samples))

        for _ in range(n_runs):
            kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', init='k-medoids++').fit(dtw_matrix)
            labels = kmedoids.labels_

            # Update co-occurrence matrix
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_occurrence_matrix[i, j] += 1

        # Normalize the co-occurrence matrix and do consensus clustering
        co_occurrence_matrix /= n_runs
        Z = linkage(co_occurrence_matrix, method='ward')
        consensus_labels = fcluster(Z, t=num_clusters, criterion='maxclust')

        # Evaluate consensus clustering using silhouette score
        score = silhouette_score(dtw_matrix, consensus_labels, metric='precomputed')
        silhouette_scores.append(score)

    return clusters_possible_range, silhouette_scores, consensus_labels


def get_optimal_clusters_plot(clusters_dict: dict,
                              clusters_possible_range: np.array,
                              silhouette_scores: np.array) -> int:
    """
    Determine the optimal number of clusters using the elbow method with silhouette scores and plot the results.

    Args:
    - clusters_dict (dict): A dictionary containing cluster labels as keys and corresponding data points as values.
    - clusters_possible_range (np.array): An array of possible cluster sizes.
    - silhouette_scores (np.array): Silhouette scores for each cluster size in clusters_possible_range.

    Returns:
    int: The optimal number of clusters determined using the elbow method.

    This function calculates the optimal number of clusters by analyzing the silhouette scores and
    plotting them to find the "elbow point" where the score starts to level off. It also compares
    the result with the true number of clusters based on the input clusters_dict.

    Example:
    optimal_clusters = get_optimal_clusters_plot(clusters_dict, clusters_possible_range, silhouette_scores)
    """

    true_number_of_clusters = len(clusters_dict.keys())
    optimal_number_of_clusters = clusters_possible_range[np.argmin(silhouette_scores)]

    # Plotting the silhouette scores to look for the "elbow"
    plt.figure(figsize=(10, 6))
    plt.plot(clusters_possible_range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Elbow Method using Silhouette Score for Consensus Clustering')
    plt.axvline(optimal_number_of_clusters,
                color='r', linestyle='--', 
                label=f'Optimal Number of Clusters: {optimal_number_of_clusters}')
    plt.axvline(27, color='green', linestyle='--', 
                label=f'True Number of Clusters: {true_number_of_clusters}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.show()

    return optimal_number_of_clusters


def get_coocurrence_matrix_plot(dtw_matrix: np.array,
                                optimal_number_of_clusters: int,
                                n_runs: int = 10000,
                                t_clusters: int = 9) -> np.array:
    """
    Generate a heatmap of the co-occurrence matrix with cluster blocks for visualization.

    Args:
    - dtw_matrix (np.array): A precomputed distance matrix.
    - optimal_number_of_clusters (int): The optimal number of clusters determined for consensus clustering.
    - n_runs (int): Number of runs for KMedoids clustering. Default is 10,000.
    - t_clusters (int): Number of clusters to extract from the co-occurrence matrix using hierarchical clustering. Default is 9.

    Returns:
    np.array: The optimal number of denoised clusters

    This function calculates a co-occurrence matrix by running KMedoids clustering, normalizes it,
    and then clusters the normalized matrix using hierarchical clustering. It visualizes the matrix
    as a heatmap with cluster blocks highlighted.

    Example:
    get_cooccurrence_matrix_plot(dtw_matrix, optimal_number_of_clusters)
    """

    # Initialize co-occurrence matrix
    n_samples = dtw_matrix.shape[0]
    co_occurrence_matrix = np.zeros((n_samples, n_samples))

    for _ in range(n_runs):
        kmedoids = KMedoids(n_clusters=optimal_number_of_clusters, metric='precomputed', init='k-medoids++').fit(dtw_matrix)
        labels = kmedoids.labels_

        # Update co-occurrence matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    co_occurrence_matrix[i, j] += 1

    # Normalize the co-occurrence matrix
    co_occurrence_matrix /= n_runs

    # Cluster the co-occurrence matrix using hierarchical clustering with Ward's method
    Z = linkage(co_occurrence_matrix, method='ward')
    consensus_labels = fcluster(Z, t=t_clusters, criterion='maxclust')

    # Sorting co-occurrence matrix by consensus_labels for better visualization
    sorted_indices = np.argsort(consensus_labels)
    sorted_matrix = co_occurrence_matrix[sorted_indices][:, sorted_indices]

    # Create a mask to overlay cluster blocks
    mask = np.ones_like(sorted_matrix)
    unique_labels = np.unique(consensus_labels)

    start_idx = 0
    for label in unique_labels:
        size = np.sum(consensus_labels == label)
        mask[start_idx:start_idx+size, start_idx:start_idx+size] = 0
        start_idx += size

    # Calculate the horizontal and vertical differences separately.
    horizontal_diff = np.diff(mask, axis=0)
    vertical_diff = np.diff(mask, axis=1)

    # Pad the shorter axis of each difference to match the original matrix size.
    horizontal_diff_padded = np.pad(horizontal_diff, ((0, 1), (0, 0)), mode='constant')
    vertical_diff_padded = np.pad(vertical_diff, ((0, 0), (0, 1)), mode='constant')

    # Combine the differences to get the borders.
    borders = horizontal_diff_padded + vertical_diff_padded

    # Plotting heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, cmap='Blues', cbar=False)  # Main heatmap
    sns.heatmap(borders, cmap='Reds', cbar=False, mask=borders == 0)  # Overlay for cluster blocks

    plt.title("Co-occurrence Matrix Heatmap with Cluster Blocks")
    plt.show()

    return consensus_labels


def get_sectors(consensus_labels: np.array) -> pd.DataFrame:
    """
    Create a DataFrame to assign sector labels to stock IDs based on consensus clustering.

    Args:
    - consensus_labels (np.array): Array of consensus cluster labels for each stock.

    Returns:
    pd.DataFrame: A DataFrame with two columns - 'stock_id' and 'sector_id'.

    This function assigns sector labels to stock IDs based on the consensus clustering results
    and returns a DataFrame that maps each stock to its corresponding sector.

    Example:
    sectors_df = get_sectors(consensus_labels)
    """

    # Create DataFrame
    stock_ids = list(range(0, len(consensus_labels)))
    data = {'stock_id': stock_ids, 'sector_id': consensus_labels}
    sectors = pd.DataFrame(data)

    return sectors