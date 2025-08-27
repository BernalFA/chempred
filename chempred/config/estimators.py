"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from typing import Literal

from chempred.config.functions import get_ml_estimators, all_estimators_in_package
from chempred.config.settings import Defaults


DEFAULT_ESTIMATORS = Defaults(
    classifiers=(
        "DummyClassifier",
        "GaussianNB",
        "GaussianProcessClassifier",
        "KNeighborsClassifier",
        "LogisticRegression",
        "MLPClassifier",
        "RandomForestClassifier",
        "RidgeClassifier",
        "SVC",
        "XGBClassifier",
        "LGBMClassifier",
    ),
    regressors=(
        "GaussianProcessRegressor",
        "KNeighborsRegressor",
        "LinearRegression",
        "MLPRegressor",
        "RandomForestRegressor",
        "Ridge",
        "SVR",
        "XGBRegressor",
        "LGBMRegressor",
    ),
    samplers=(
        "RandomUnderSampler",
        "SMOTE",
    )
)


def get_available_estimators(
        category: Literal["classifiers", "regressors", "samplers", "mol_transformers"]
) -> list[tuple]:
    """Provide a list of estimators according to the given category.

    Args:
        category (Literal['classifiers', 'regressors', 'samplers', 'mol_transformers']):
                    type of estimator.

    Returns:
        list[tuple]: available estimators as name : Callable pairs.
    """
    if category in ["classifiers", "regressors"]:
        default_names = getattr(DEFAULT_ESTIMATORS, category)
        estimators = get_ml_estimators()
        estimators = [est for est in estimators if est[0] in default_names]
    elif category == "samplers":
        default_names = getattr(DEFAULT_ESTIMATORS, category)
        estimators = all_estimators_in_package("imblearn")
        estimators = [est for est in estimators if est[0] in default_names]
    elif category == "mol_transformers":
        estimators = all_estimators_in_package("scikit_mol")
    return estimators


CLASSIFIERS = get_available_estimators("classifiers")

REGRESSORS = get_available_estimators("regressors")

SAMPLING_METHODS = get_available_estimators("samplers")

MOL_TRANSFORMERS = get_available_estimators("mol_transformers")
