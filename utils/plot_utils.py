import matplotlib.pyplot as plt
from utils.monte_carlo_utils import get_gbm_with_seasonality_paths


def plot_data_sample(clusters_dict: dict) -> None:
    """
    Plot data from a dictionary of clusters.

    This function takes a dictionary containing cluster data and generates subplots
    to visualize the data. It groups clusters into 'low_ret', 'med_ret', and 'high_ret'
    categories and arranges them in a 3x3 grid of subplots with appropriate legends.

    Parameters:
    clusters_dict (dict): A dictionary containing cluster data where keys represent
                         cluster names, and values are dictionaries with cluster parameters.

    Returns:
    None: This function displays the generated plots but does not return any values.

    Example:
    clusters_dict = {
        'low_ret_low_vol_slow_season': {...},
        'low_ret_low_vol_med_season': {...},
        # ... (other cluster data)
    }
    plot_data_sample(clusters_dict)
    """

    # Initialize a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(25, 15))  # 9 rows, 3 columns for 27 plots

    # Create dictionaries to group keys by 'low_ret', 'med_ret', and 'high_ret'
    ret_categories = {'low_ret': [], 'med_ret': [], 'high_ret': []}

    # Iterate over dictionary keys and group them
    for key in clusters_dict.keys():
        if 'low_ret' in key:
            ret_categories['low_ret'].append(key)
        elif 'med_ret' in key:
            ret_categories['med_ret'].append(key)
        elif 'high_ret' in key:
            ret_categories['high_ret'].append(key)

    # Iterate over the categories and keys to generate and plot the data
    for i, category in enumerate(['low_ret', 'med_ret', 'high_ret']):
        for j, key in enumerate(ret_categories[category]):
            cluster_temp = clusters_dict[key]
            
            # Call your function to get data and plot
            paths = get_gbm_with_seasonality_paths(N_sim=1,
                                                mu=cluster_temp['mu'],
                                                sigma=cluster_temp['sigma'],
                                                alpha=cluster_temp['alpha'],
                                                T=1.0,
                                                n_steps=100)
            
            # Add the plot to the appropriate subplot
            row = j // 3
            col = i  # Use i to keep the category in the specified column
            
            # Replace 'plot_data' with the appropriate function or code to plot your data
            # For example, if 'plot' contains the Matplotlib plot object, you can do:
            axes[row, col].plot(paths.T, label=key)  # Add label to each plot
            
            # Set subplot title using the key from the dictionary
            axes[row, col].set_title('_'.join(key.split('_')[:4]))
            
            # Add a legend to each subplot
            axes[row, col].legend(loc='upper left')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show or save the final figure
    plt.show()