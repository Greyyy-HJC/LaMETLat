"""
Setup standard logging and cache output.
"""

import os
import logging
import sys

def set_my_logger(log_file, console_output=False):
    """
    Set up a custom logger.

    Parameters:
    log_file (str): The path to the log file.
    console_output (bool): Whether to also output logs to console. Default is False.
    """
    # Delete existing log_file
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Create a new log_file
    open(log_file, "w").close()
    
    # Create a logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

