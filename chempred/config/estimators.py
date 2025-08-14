"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators

from chempred.utils import all_estimators_in_package


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
