"""
Sklearn compatible transformers for removal of correlated features up to defined
threshold and removal of missing values.

@author: Dr. Freddy A. Bernal
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


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
        # first validate data
        X = validate_data(self, X, ensure_min_features=2)
        # Define n_features
        self.n_features_in_ = X.shape[1]
        # Calculate pairwise correlations
        self.correlations_ = np.corrcoef(X, rowvar=False)

        return self


class MissingValuesRemover(TransformerMixin, BaseEstimator):
    """Sklearn compatible transformer to remove features containing missing or infinite
    values.
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold  # For future implementation based on threshold

    def fit(self, X, y=None):
        # first validate data
        # X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)
        X = self._check_data_validity(X)
        # Define n_features and training samples
        self.n_features_in_ = X.shape[1]
        self.n_train_samples_ = X.shape[0]
        # Define NaNs and inf in training data (features instead of compounds)
        self.is_finite = np.isfinite(X).all(axis=0)

        return self

    def transform(self, X):
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # validate data
        # X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)
        X = self._check_data_validity(X)
        # Remove NaN and Inf
        X = X[:, self.is_finite]
        # in case of test data, check for additional missing or infinite values
        if not np.isfinite(X).all():
            # Remove compounds with conflicting values (NaN or Inf)
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
        return X

    def _check_data_validity(self, X):
        """Check if the data set exclusively consists of missing or infinite values.

        Args:
            X (np.ndarray): Input data to check.

        Returns:
            np.ndarray: dataset if not completely invalid.
        """
        not_finite = ~np.isfinite(X)
        if not_finite.sum() == X.size:
            raise ValueError("Data set contains only missing and/or infinite values.")
        else:
            X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)

        return X
