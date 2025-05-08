"""
Sklearn compatible transformers for removal of correlated features up to defined
threshold and removal of missing values.

@author: Dr. Freddy A. Bernal
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class RemoveCorrelated(SelectorMixin, BaseEstimator):
    """Sklearn compatible transformer to remove highly correlated features
    from given dataset.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold (float, optional): minimum value to consider two features
                                         highly correlated. Defaults to 0.8.
        """
        self.threshold = threshold

    def _get_support_mask(self):
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # Check for correlations >= threshold
        mask = np.full((self.correlations_.shape[0],), True, dtype=bool)
        for i in range(self.correlations_.shape[0]):
            for j in range(i + 1, self.correlations_.shape[0]):
                if abs(self.correlations_[i, j]) >= self.threshold:
                    if mask[j]:
                        mask[j] = False

        return mask

    def fit(self, X, y=None):
        # Define n_features
        self.n_features_in_ = X.shape[1]
        # Calculate pairwise correlations
        if self.n_features_in_ > 1:
            self.correlations_ = np.corrcoef(X, rowvar=False)
        else:
            self.correlations_ = np.array([1])

        return self


class MissingValuesRemover(TransformerMixin, BaseEstimator):
    """Sklearn compatible transformer to remove features containing missing or infinite
    values.
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold  # For future implementation based on threshold

    def fit(self, X, y=None):
        # Define n_features and training samples
        self.n_features_in_ = X.shape[1]
        self.n_train_samples_ = X.shape[0]
        # Define NaNs in training data (features instead of compounds)
        self.is_nan = np.isnan(X).any(axis=0)
        # Define Inf in training data (features instead of compounds)
        self.is_inf = np.isinf(X).any(axis=0)

        return self

    def transform(self, X):
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # Remove NaN and Inf
        X = X[:, (~self.is_nan) & (~self.is_inf)]
        # in case of test data, check for additional missing values
        if X.shape[0] != self.n_train_samples_:
            mask1 = np.isnan(X).any(axis=1)
            mask2 = np.isinf(X).any(axis=1)
            X = X[(~mask1) & (~mask2)]
        return X
