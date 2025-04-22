"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""

from dataclasses import dataclass
from typing import Optional

from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
from utils import all_estimators_in_package


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
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin)) and (est[0] in selected_classifiers)
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


@dataclass
class SimpleConfig:
    classifier: tuple
    sampler: Optional[tuple] = None
    transformer: Optional[tuple] = None
