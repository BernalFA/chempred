"""
Module with a helper functions for varied uses.

@author: Dr. Freddy A. Bernal
"""
import subprocess
import time
from functools import wraps

import pandas as pd


def add_timing(func):
    """Decorator to add execution time measurements to results from func"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        result.update({"Time": execution_time})
        return result

    return wrapper


def get_url():
    remote_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"],
        text=True
    ).strip()
    url_repo = remote_url.replace(
        "git@github.com:", "https://raw.githubusercontent.com/"
    ).removesuffix(".git")
    path = "/".join(["refs", "heads", "main", "tests", "data"])
    url_data = url_repo + "/" + path
    return url_data


def load_classification_data():
    url_data = get_url()
    return pd.read_csv(url_data + "/BBBP.csv")


def load_regression_data():
    url_data = get_url()
    return pd.read_csv(url_data + "/Lipophilicity.csv").sample(1982, random_state=21)
