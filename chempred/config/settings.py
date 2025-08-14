"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""

from dataclasses import dataclass
from typing import Optional

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
)
from sklearn.utils import all_estimators

from chempred.utils import all_estimators_in_package
from chempred.metrics import prc_auc_score, f0_5_score, f2_score, ef_score


selected_classifiers = [
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
]

_SKLEARN_CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin)) and (est[0] in selected_classifiers)
]

_OTHER_CLASSIFIERS = all_estimators_in_package("lightgbm")
_OTHER_CLASSIFIERS.extend(all_estimators_in_package("xgboost"))
_OTHER_CLASSIFIERS = [
    est
    for est in _OTHER_CLASSIFIERS
    if est[0] in selected_classifiers
]

CLASSIFIERS = _SKLEARN_CLASSIFIERS + _OTHER_CLASSIFIERS

selected_regressors = [
    "GaussianProcessRegressor",
    "KNeighborsRegressor",
    "LinearRegression",
    "MLPRegressor",
    "RandomForestRegressor",
    "Ridge",
    "SVR",
    "XGBRegressor",
    "LGBMRegressor",
]

_SKLEARN_REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin)) and (est[0] in selected_regressors)
]

_OTHER_REGRESSORS = all_estimators_in_package("lightgbm")
_OTHER_REGRESSORS.extend(all_estimators_in_package("xgboost"))
_OTHER_REGRESSORS = [
    est
    for est in _OTHER_REGRESSORS
    if est[0] in selected_regressors
]

REGRESSORS = _SKLEARN_REGRESSORS + _OTHER_REGRESSORS

selected_sampling_methods = [
    "RandomUnderSampler",
    "SMOTE",
]

SAMPLING_METHODS = [
    est
    for est in all_estimators_in_package("imblearn")
    if est[0] in selected_sampling_methods
]

MOL_TRANSFORMERS = all_estimators_in_package("scikit_mol")


@dataclass
class SimpleConfig:
    estimator: tuple
    sampler: Optional[tuple] = None
    transformer: Optional[tuple] = None


CLASSIFICATION_SCORERS = dict(
    balanced_accuracy=balanced_accuracy_score,
    f1=f1_score,
    roc_auc=roc_auc_score,
    prc_auc=prc_auc_score,
    mcc=matthews_corrcoef,
    cohen_kappa=cohen_kappa_score,
    f05=f0_5_score,
    f2=f2_score,
    recall=recall_score,
    precision=precision_score,
    ef=ef_score,
)

REGRESSION_SCORERS = dict(
    r2=r2_score,
    mae=mean_absolute_error,
    mse=mean_squared_error,
    rmse=root_mean_squared_error
)


def get_scorer_names() -> list:
    """Get names of available scorers for model evaluation.

    Returns:
        list: Names of available scorers.
    """
    return sorted(CLASSIFICATION_SCORERS.keys())
