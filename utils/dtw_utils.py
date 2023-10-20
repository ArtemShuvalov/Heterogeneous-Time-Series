import pandas as pd
import numpy as np
from fastdtw import fastdtw

from utils.calc_utils import get_time

def get_l_n_distance(x: np.array,
                     y: np.array,
                     l_norm: str = 'L1') -> np.array:
    """
    Calculate the L1 or L2 distance between two arrays x and y.

    Parameters:
    - x (np.array): The first input array.
    - y (np.array): The second input array, of the same shape as x.
    - l_norm (str, optional): The norm used for distance calculation. 
      Valid values are 'L1' (default) for L1 (Manhattan) distance and 'L2' for L2 (Euclidean) distance.

    Returns:
    - distance (np.array): The calculated distance between x and y based on the specified norm.

    This function computes the distance between two arrays x and y based on the specified norm.
    - If l_norm is 'L1', it calculates the L1 (Manhattan) distance as the sum of absolute differences.
    - If l_norm is 'L2', it calculates the L2 (Euclidean) distance as the square root of the sum of squared differences.

    Example usage:
    l1_distance = get_l_n_distance(array1, array2, l_norm='L1')
    l2_distance = get_l_n_distance(array1, array2, l_norm='L2')
    
    """
    
    if l_norm == 'L1':
        distance = np.sum(np.abs(x - y))
    elif l_norm == 'L2':
        distance = np.sqrt(np.sum((x - y) ** 2))
    else:
        print('Input Correct Distance Measure: L1, L2')

    return distance


def get_dtw_matrix(input_df: pd.DataFrame,
                   save_data: bool = True,
                   filename: str = 'DtwMatrix') -> np.array:
    """
    Calculate the Dynamic Time Warping (DTW) distance matrix for a DataFrame of stock data.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing stock data.
    - save_data (bool, optional): Whether to save the DTW matrix as a CSV file (default is True).
    - filename (str, optional): The base filename for saving the DTW matrix as a CSV file (default is 'DtwMatrix').

    Returns:
    - dtw_matrix (np.array): The computed DTW distance matrix.

    This function computes the DTW distance between each pair of time series in the input DataFrame
    and stores the distances in a matrix. The matrix is symmetric, where dtw_matrix[i, j] represents
    the DTW distance between stock_id i and stock_id j.

    If save_data is set to True, the function saves the DTW matrix as a CSV file with a timestamp.

    Note: Ensure you have the necessary modules and functions imported for 'get_time' and 'get_l_n_distance'.

    Example usage:
    dtw_matrix = get_dtw_matrix(stock_data, save_data=True, filename='MyDtwMatrix')

    """
    
    n = input_df.shape[1]
    dtw_matrix = np.zeros((n, n))

    for i in range(n):
        if i % 50 == 0:
            print(f'{get_time()}: Work on stock_id={i}')
        # Compute Upper Triangule Values
        for j in range(i+1, n): 
            distance, _ = fastdtw(input_df[i].values,
                                  input_df[j].values, 
                                  dist=get_l_n_distance)
            dtw_matrix[i, j] = distance
            # Copy to Lower Triangle
            dtw_matrix[j, i] = distance 
    
    if save_data:
        time_now = get_time(format=None).strftime('%Y%m%d_%H%M%S')
        np.savetxt(f'{filename}_{time_now}.csv', dtw_matrix, delimiter=',')

    return dtw_matrix