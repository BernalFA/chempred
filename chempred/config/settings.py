"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
)

from chempred.metrics import prc_auc_score, f0_5_score, f2_score, ef_score


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
