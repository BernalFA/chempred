"""
Sklearn compatible transformer for removal of correlated features up to defined
threshold.

@author: Dr. Freddy A. Bernal
"""

import numpy as np
from sklearn.base import BaseEstimator
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
        self.correlations_ = np.corrcoef(X, rowvar=False)

        return self
