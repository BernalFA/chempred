"""
Module to define available evaluation metrics derived from sklearn.

@author: Dr. Freddy A. Bernal
"""
import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve, auc, fbeta_score


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
