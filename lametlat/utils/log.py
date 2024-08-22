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
