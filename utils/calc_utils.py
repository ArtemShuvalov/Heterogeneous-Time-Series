from datetime import datetime

def get_time(format: str = '%H:%M:%S') -> datetime:
    """
    Get the current time as a datetime object or formatted string.

    Parameters:
    - format (str, optional): The desired format for the time as a string. 
      Default format is '%H:%M:%S' (hour:minute:second).

    Returns:
    - output_time (datetime or str): The current time as a datetime object or formatted string.

    This function retrieves the current time and returns it either as a datetime object or as a formatted string
    based on the specified format. If no format is provided, it defaults to '%H:%M:%S' (hour:minute:second).

    Example usage:
    current_time = get_time()  # Returns a datetime object representing the current time.
    formatted_time = get_time(format='%Y-%m-%d %H:%M:%S')  # Returns a formatted time string.

    """

    if format:
        output_time = datetime.now().strftime(format)
    else:
        output_time = datetime.now()
    
    return output_time