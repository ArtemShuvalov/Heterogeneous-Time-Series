import pandas as pd
import numpy as np
from fastdtw import fastdtw

from utils.calc_utils import get_time

def get_l_n_distance(x: np.array,
                     y: np.array,
                     l_norm: str = 'L1') -> np.array:
    
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


