"""
Module with a custom function to read all available estimators in a selected dependency
and a decorator able to add execution time to an array.

@author: Dr. Freddy A. Bernal
"""

import inspect
import pkgutil
import time
from functools import wraps
from importlib import import_module
from operator import itemgetter


PATH = "/home/freddy/miniconda3/envs/chempred/lib/python3.12/site-packages/"


_MODULE_TO_IGNORE = ["tests", "base", "plotting"]


def add_timing(func):
    """Decorator to add time execution measurements to results from func"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result.tolist() + [execution_time]

    return wrapper


def all_estimators_in_package(package):
    # adapted from sklearn all_estimators function

    def is_abstract(c):
        if not hasattr(c, "__abstractmethods__"):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    for _, module_name, _ in pkgutil.walk_packages(
        path=[PATH + package],
        prefix=package + "."
    ):
        module_parts = module_name.split(".")
        if (
            any(part in _MODULE_TO_IGNORE for part in module_parts)
            or "._" in module_name
            or "test" in module_name
        ):
            continue
        module = import_module(module_name)
        classes = inspect.getmembers(module, inspect.isclass)
        classes = [
            (name, est_cls)
            for name, est_cls in classes
            if not name.startswith("_")
        ]
        all_classes.extend(classes)

    all_classes = set(all_classes)
    estimators = filter_classes(all_classes, pkg=package)
    estimators = [c for c in estimators if not is_abstract(c[1])]

    return sorted(set(estimators), key=itemgetter(0))


def filter_classes(all_classes, pkg):
    if pkg == "imblearn":
        from imblearn.base import BaseSampler

        estimators = [
            c for c in all_classes
            if (issubclass(c[1], BaseSampler) and c[0] != "BaseSampler")
        ]

    elif pkg == "scikit_mol":
        from scikit_mol.fingerprints.baseclasses import BaseFpsTransformer

        estimators = [
            c
            for c in all_classes
            if (
                issubclass(c[1], BaseFpsTransformer)
                and c[0] != "BaseFpsTransformer"
                and c[0] != "cls"
            )
            or (c[0] == "MolecularDescriptorTransformer")
        ]

    return estimators
