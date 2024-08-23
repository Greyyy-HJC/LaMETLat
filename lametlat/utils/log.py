"""
Setup standard logging and cache output.
"""

import os
import logging

def set_up_log(log_file):
    """
    Set up logging to a file.

    Parameters:
    log_file (str): The path to the log file.

    Returns:
    None
    """
    # make sure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if not os.path.exists(log_file):
        print(f"Creating new log file: {log_file}")
    
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    