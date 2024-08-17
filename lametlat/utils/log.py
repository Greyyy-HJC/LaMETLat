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
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    global fit_count, bad_fit_count
    fit_count = 0
    bad_fit_count = 0

def log_count_fit(message):
    """
    Increments the fit_count and bad_fit_count variables and logs a message if provided.

    Parameters:
    message (str): The bad fit message to be logged.

    Returns:
    None
    """
    global fit_count, bad_fit_count
    fit_count += 1

    if message:
        bad_fit_count += 1
        logging.info(message)