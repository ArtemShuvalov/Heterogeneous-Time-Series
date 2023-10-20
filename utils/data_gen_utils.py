import pandas as pd
import numpy as np

def generate_clusters_dict(n_rets: int = 3, 
                           n_vols: int = 3, 
                           n_season: int = 3, 
                           ret_base: float = 0.05, 
                           vol_base: float = 0.1, 
                           season_base: float = 0.5) -> dict:
    """
    Generate a dictionary of clusters with parameter combinations.

    Parameters:
    - n_rets (int, optional): Number of return levels (low, medium, high). Default is 3.
    - n_vols (int, optional): Number of volatility levels (low, medium, high). Default is 3.
    - n_season (int, optional): Number of seasonality levels (slow, medium, fast). Default is 3.
    - ret_base (float, optional): Base value for returns. Default is 0.05.
    - vol_base (float, optional): Base value for volatility. Default is 0.1.
    - season_base (float, optional): Base value for seasonality. Default is 0.5.

    Returns:
    - dict: A dictionary of clusters with parameter combinations, including cluster IDs, mu, sigma, and alpha.

    Note:
    The function generates clusters with unique combinations of return, volatility, and seasonality levels.
    Each cluster includes a cluster ID and corresponding parameter values (mu, sigma, and alpha).

    """

    # Generate dictionary with clusters
    clusters_dict = {
                    f'{"low" if ret_temp == 0 else "med" if ret_temp == 1 else "high"}_ret_'
                    f'{"low" if vol_temp == 0 else "med" if vol_temp == 1 else "high"}_vol_'
                    f'{"slow" if season_temp == 0 else "med" if season_temp == 1 else "fast"}_season': 
                    {'cluster_id': ret_temp * n_vols * n_season + vol_temp * n_season + season_temp,
                    'mu': np.round(ret_base * (ret_temp + 1), 4),
                    'sigma': np.round(vol_base * (vol_temp + 1), 4),
                    'alpha': np.round(season_base * (season_temp + 1), 4)}
                    for ret_temp in range(n_rets)
                    for vol_temp in range(n_vols)
                    for season_temp in range(n_season)
                   }
    
    return clusters_dict


def get_sectors_df(df_prices: pd.DataFrame,
                   df_sectors_map: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a DataFrame representing sectors based on stock prices and sector mapping.

    Parameters:
    - df_prices (pd.DataFrame): The DataFrame containing stock prices with stock IDs as columns.
    - df_sectors_map (pd.DataFrame): The DataFrame mapping stock IDs to sector IDs.

    Returns:
    - df_sectors (pd.DataFrame): A DataFrame representing sectors with sector IDs as columns
      and the sector's mean stock prices over time.

    This function creates a DataFrame representing sectors based on stock prices and a mapping
    of stock IDs to sector IDs. For each sector, it calculates the mean stock prices of all
    stocks belonging to that sector over time.

    Example usage:
    sector_prices = get_sectors_df(stock_prices_df, sector_mapping_df)
    """
    
    df_sectors = pd.DataFrame(index=df_prices.index, 
                              columns=df_prices.columns)

    for sector_id_temp in df_sectors_map.sector_id:
        stock_id_arr_temp = df_sectors_map[df_sectors_map.sector_id == sector_id_temp].stock_id

        for stock_id_temp in stock_id_arr_temp:
            df_sectors[stock_id_temp] = df_prices[stock_id_arr_temp].mean(axis=1)

    return df_sectors