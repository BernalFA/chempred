"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""

from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef
)
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


def prc_auc_score(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Calculate PRC AUC for given dataset.

    Args:
        y_true (npt.ArrayLike): true labels.
        y (npt.ArrayLike): target scores as probabilities (from predict_proba or
                           decision_function in sklearn estimators).

    Returns:
        float: PRC AUC value
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


SCORERS = dict(
    balanced_accuracy=balanced_accuracy_score,
    f1=f1_score,
    roc_auc=roc_auc_score,
    prc_auc=prc_auc_score,
    mcc=matthews_corrcoef,
)


def get_scorer_names() -> list:
    """Get names of available scorers for model evaluation.

    Returns:
        list: Names of available scorers.
    """
    return sorted(SCORERS.keys())
