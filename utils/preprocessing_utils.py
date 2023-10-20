import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Min-Max scaling for each time series
def get_min_max_scaled_values(input_df: pd.DataFrame,
                              feature_min: float = 0,
                              feature_max: float = 1) -> pd.DataFrame:
    """
    Perform Min-Max scaling on each time series in a DataFrame with custom feature range.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing time series data.
    - feature_min (float, optional): The minimum value to which the features should be scaled (default is 0).
    - feature_max (float, optional): The maximum value to which the features should be scaled (default is 1).

    Returns:
    - df_scaled (pd.DataFrame): The DataFrame with time series data scaled to the specified feature range.

    This function applies Min-Max scaling to each time series within the input DataFrame, scaling each
    feature to the specified custom feature range defined by 'feature_min' and 'feature_max'.

    Example usage:
    scaled_data = get_min_max_scaled_values(stock_data, feature_min=0, feature_max=100)
    
    """
    
    scaler = MinMaxScaler(feature_range=(feature_min, 
                                         feature_max))
    
    df_scaled = pd.DataFrame(index=input_df.index,
                             columns=input_df.columns)
    for stock_id in input_df.columns:
        data_temp = scaler.fit_transform(input_df[stock_id].values.reshape(-1, 1))
        df_scaled[stock_id] = data_temp
    
    return df_scaled