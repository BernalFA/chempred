"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from dataclasses import dataclass

from chempred.config.functions import get_ml_estimators, all_estimators_in_package


@dataclass(frozen=True)
class Defaults:
    classifiers: tuple[str]
    regressors: tuple[str]
    samplers: tuple[str]


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


CLASSIFIERS = [
    est
    for est in get_ml_estimators("classification")
    if est[0] in DEFAULT_ESTIMATORS.classifiers
]


REGRESSORS = [
    est
    for est in get_ml_estimators("regression")
    if est[0] in DEFAULT_ESTIMATORS.regressors
]


SAMPLING_METHODS = [
    est
    for est in all_estimators_in_package("imblearn")
    if est[0] in DEFAULT_ESTIMATORS.samplers
]

MOL_TRANSFORMERS = all_estimators_in_package("scikit_mol")
