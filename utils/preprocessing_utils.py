import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Min-Max scaling for each time series
def get_min_max_scaled_values(input_df: pd.DataFrame) -> pd.DataFrame:
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    df_scaled = pd.DataFrame(index=input_df.index,
                             columns=input_df.columns)
    for stock_id in input_df.columns:
        data_temp = scaler.fit_transform(input_df[stock_id].values.reshape(-1, 1))
        df_scaled[stock_id] = data_temp
    
    return df_scaled