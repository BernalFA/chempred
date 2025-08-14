"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from chempred.config.functions import get_ml_estimators, all_estimators_in_package


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


CLASSIFIERS = [
    est
    for est in get_ml_estimators("classification")
    if est[0] in selected_classifiers
]

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

REGRESSORS = [
    est
    for est in get_ml_estimators("regression")
    if est[0] in selected_regressors
]

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
