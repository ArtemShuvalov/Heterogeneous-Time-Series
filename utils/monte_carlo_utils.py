import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_gbm_with_seasonality_draw(mu: float,
                                  sigma: float,
                                  T: float,
                                  S0: float = 100,
                                  alpha: float = 0,
                                  n_steps: int = 252,
                                  random_seed: int = 0) -> np.ndarray:
    """
    Simulate a Geometric Brownian Motion (GBM) with seasonalities.

    Parameters:
    - mu (float): Drift (average rate of return).
    - sigma (float): Volatility (standard deviation of returns).
    - T (float): Time horizon in years.
    - S0 (float, optional): Initial asset price. Default is 100.
    - alpha (float, optional): Daily seasonal factor. Default is 0.
    - n_steps (int, optional): Number of steps in the simulation. Default is 252.
    - random_seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
    - np.ndarray: An array of simulated asset prices over time.

    """
    
    # Set random seed for consistency of results
    np.random.seed(random_seed)

    # Define discretization step
    dt = T / n_steps
    # Define time vector
    t = np.linspace(0, T, n_steps)

    S = np.zeros(n_steps)
    S[0] = S0
    # Daily seasonal sinusoidal factor
    alpha = np.sin(2 * np.pi * t * alpha)  
    
    # Create stock values
    for i in range(1, n_steps):
        dW = np.random.normal(0, 1) * np.sqrt(dt)
        S[i] = S[i-1] * (1 + (mu + alpha[i-1]) * dt + sigma * dW)
    
    return S


def get_gbm_with_seasonality_paths(N_sim: int = 100, 
                                   plot_data: bool = False,
                                   return_plot: bool = False,
                                   **kwargs) -> np.ndarray:
    """
    Generate multiple paths of GBM with seasonalities.

    Parameters:
    - N_sim (int, optional): Number of paths to generate. Default is 100.
    - plot_data (bool, optional): Variable governing if we want to see plot. Default is False.
    - return_data (bool, optional): Variable governing if we want to return plot. Default is False.
    - **kwargs: Additional arguments to pass to get_gbm_with_seasonality_draw.

    Returns:
    - np.ndarray: A 2D array representing multiple simulated paths of asset prices.

    """
        
    # Get parameters from kwargs
    n_steps = kwargs.get('n_steps') 
    T = kwargs.get('T')
    mu = kwargs.get('mu')
    sigma = kwargs.get('sigma')
    alpha = kwargs.get('alpha')
    
    t = np.linspace(0, T, n_steps)
    
    # Create vector for paths
    paths = np.ndarray(shape=(N_sim, n_steps))
    
    # Iterate through number of simulations
    for i in range(N_sim):
        path_temp = get_gbm_with_seasonality_draw(**kwargs)
        paths[i, :] = path_temp
    
    if plot_data:
        # Plot the simulated paths
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               figsize=(10, 6))
        ax.plot(t, paths.T)
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'GBM with Seasonalities: $\\mu$={mu}, $\\sigma$={sigma}, $\\alpha$={alpha}')
        
        plt.show()
    
    if return_plot:
        return paths, fig
    else:
        return paths
    

def get_stocks_universe(clusters_dict: dict,
                        stocks_per_cluster: int = 10,
                        **kwargs) -> pd.DataFrame:
    """
    Generate a DataFrame representing a universe of stocks based on clusters.

    This function generates a DataFrame that represents a universe of stocks by
    simulating multiple stock paths for each cluster in the given dictionary
    (clusters_dict). Each cluster defines the parameters for generating stock paths,
    and 'stocks_per_cluster' determines the number of stock paths to generate for
    each cluster. The resulting DataFrame contains stock paths with corresponding
    stock IDs.

    Parameters:
    clusters_dict (dict): A dictionary containing cluster data where keys represent
                         cluster names, and values are dictionaries with cluster parameters.
    stocks_per_cluster (int, optional): Number of stock paths to generate per cluster.
                                        Default is 10.
    **kwargs: Additional arguments to pass to 'get_gbm_with_seasonality_paths'.

    Returns:
    pd.DataFrame: A DataFrame with stock paths representing a universe of stocks.
                  Each column contains the stock price paths, and the column names
                  represent stock IDs.

    Example:
    clusters_dict = {
        'low_ret_low_vol_slow_season': {...},
        'low_ret_low_vol_med_season': {...},
        # ... (other cluster data)
    }
    universe_df = get_stocks_univers(clusters_dict, 
                                     stocks_per_cluster=10, 
                                     T=1.0, n_steps=100)
    """

    T = kwargs.get('T')
    n_steps = kwargs.get('n_steps')
    
    # Initialize an empty DataFrame to store the data
    df = pd.DataFrame()

    # Iterate over the keys in clusters_dict
    for stock_id, key in enumerate(clusters_dict.keys()):
        cluster_temp = clusters_dict[key]
        
        # Generate 'stocks_per_cluster' stock paths for each key
        paths_temp = get_gbm_with_seasonality_paths(N_sim=stocks_per_cluster,
                                                    mu=cluster_temp['mu'],
                                                    sigma=cluster_temp['sigma'],
                                                    alpha=cluster_temp['alpha'],
                                                    T=T,
                                                    n_steps=n_steps)
        
        # Create a DataFrame with stock IDs and paths
        columns_temp = np.arange(0, stocks_per_cluster) + stocks_per_cluster * stock_id
        df_temp = pd.DataFrame(paths_temp.T,
                                columns=columns_temp)
        
        # Append the stock data to the main DataFrame
        df = pd.concat([df, df_temp], axis=1)

    df.index.name = 'time'
    df.columns.name = 'stock_id'
    
    return df
