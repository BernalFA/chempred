"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score, fbeta_score, precision_score, recall_score
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


def f0_5_score(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calculate f0.5 score for given dataset.

    Args:
        y_true (npt.ArrayLike): true labels.
        y (npt.ArrayLike): predicted target values returned by classifier.

    Returns:
        float: f0.5 score
    """
    return fbeta_score(y_true, y_pred, beta=0.5)


def f2_score(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calculate f2 score for given dataset.

    Args:
        y_true (npt.ArrayLike): true labels.
        y (npt.ArrayLike): predicted target values returned by classifier.

    Returns:
        float: f2 score
    """
    return fbeta_score(y_true, y_pred, beta=2)


def ef_score(
        y_true: npt.ArrayLike, y_pred: npt.ArrayLike, fraction: float = 0.05
) -> float:
    """Calculate enrichment factor for a selected fraction set.

    Args:
        y_true (npt.ArrayLike): true labels.
        y_pred (npt.ArrayLike): predicted target values returned by classifier.
        fraction (float, optional): fraction considered for evaluation.
                                    Defaults to 0.05 (5 %).

    Returns:
        float: enrichment factor for selected fraction.
    """
    # Combine arrays and sort values (descending)
    values = np.vstack((y_true, y_pred)).T
    values = values[values[:, 1].argsort()[::-1]]
    # define totals and actives
    total_compounds = len(values)
    total_actives = len(values[values[:, 1] == 1])
    # Check actives were predicted
    if total_actives > 0:
        num_set = int(total_compounds * fraction)
        selected_set = values[:num_set]
        actives_set = len(selected_set[selected_set[:, 1] == 1])
        return actives_set * total_compounds / (total_actives * num_set)
    return 0.0


SCORERS = dict(
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


def get_scorer_names() -> list:
    """Get names of available scorers for model evaluation.

    Returns:
        list: Names of available scorers.
    """
    return sorted(SCORERS.keys())
