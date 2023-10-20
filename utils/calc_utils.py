from datetime import datetime, timedelta

def get_time(format: str = '%H:%M:%S') -> datetime:
    
    if format:
        output_time = datetime.now().strftime(format)
    else:
        output_time = datetime.now()
    
    return output_time