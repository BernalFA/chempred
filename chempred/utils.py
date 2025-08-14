"""
Module with a helper functions for varied uses.

@author: Dr. Freddy A. Bernal
"""
import time
from functools import wraps


def add_timing(func):
    """Decorator to add execution time measurements to results from func"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result.tolist() + [execution_time]

    return wrapper
