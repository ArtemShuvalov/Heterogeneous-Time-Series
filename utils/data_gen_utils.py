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