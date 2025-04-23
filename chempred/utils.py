"""
Module with a custom function to read all available estimators in a selected dependency
and a decorator able to add execution time to an array.

@author: Dr. Freddy A. Bernal
"""

import inspect
import os
import pkgutil
import site
import time
from functools import wraps
from importlib import import_module
from operator import itemgetter
from typing import Literal


PATH = site.getsitepackages()[0]


_MODULE_TO_IGNORE = ["tests", "base", "plotting"]


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


def all_estimators_in_package(
        package: Literal["scikit_mol", "imblearn"]
) -> list[tuple]:
    """Search estimators available in specified package. The package must be installed
    in the environment/python distribution running the module. This function is adapted
    from sklearn all_estimators function.

    Args:
        package (Literal["scikit_mol", "imblearn"]): package name

    Returns:
        list[tuple]: available estimators in package as tuples (name, estimator).
    """

    def is_abstract(c):
        if not hasattr(c, "__abstractmethods__"):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    for _, module_name, _ in pkgutil.walk_packages(
        path=[os.path.join(PATH, package)],
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


def filter_classes(all_classes: list[tuple], pkg: str) -> list[tuple]:
    """remove unnecessary classes from general list of available classes according to
    key inheritance (based on the package itself).

    Args:
        all_classes (list[tuple]): full list of classes in package
        pkg (str): package name

    Returns:
        list[tuple]: filtered list of classes of interest
    """

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
