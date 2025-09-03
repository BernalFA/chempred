"""
Sklearn compatible transformers for removal of correlated features up to defined
threshold and removal of missing values.

@author: Dr. Freddy A. Bernal
"""

import numpy as np
import numpy.typing as npt
from descriptastorus.descriptors import dists
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted, validate_data


class RemoveCorrelated(SelectorMixin, BaseEstimator):
    """Sklearn compatible transformer to remove highly correlated features
    from given dataset.

    Example:
        ```python
        remover = RemoveCorrelated(threshold=0.8)
        X_processed = remover.fit_transform(X)
        ```
    """

    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold (float, optional): minimum value to consider two features
                                         highly correlated. Defaults to 0.8.
        """
        self.threshold = threshold

    def _get_support_mask(self) -> npt.ArrayLike:
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

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        # first validate data
        X = validate_data(self, X, ensure_min_features=2)
        # Define n_features
        self.n_features_in_ = X.shape[1]
        # Calculate pairwise correlations
        self.correlations_ = np.corrcoef(X, rowvar=False)

        return self

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)


class MissingValuesRemover(SelectorMixin, BaseEstimator):
    """Sklearn compatible transformer to remove features containing missing or infinite
    values.

    Example:
        ```python
        remover = MissingValuesRemover()
        X_processed = remover.fit_transform(X)
        ```
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold  # For future implementation based on threshold

    def _get_support_mask(self) -> npt.ArrayLike:
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        return self.is_finite

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        # first validate data
        # X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)
        X = self._check_data_validity(X)
        # Define n_features and training samples
        self.n_features_in_ = X.shape[1]
        self.n_train_samples_ = X.shape[0]
        # Define NaNs and inf in training data (features instead of compounds)
        self.is_finite = np.isfinite(X).all(axis=0)

        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # validate data
        # X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)
        X = self._check_data_validity(X)
        # Remove NaN and Inf
        if hasattr(X, "iloc"):
            X = X.iloc[:, self.is_finite]
        else:
            X = X[:, self.is_finite]
        # in case of test data, check for additional missing or infinite values
        if not np.isfinite(X).all():
            # Remove compounds with conflicting values (NaN or Inf)
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
        return X

    def _check_data_validity(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Check if the data set exclusively consists of missing or infinite values.

        Args:
            X (npt.ArrayLike): Input data to check.

        Returns:
            npt.ArrayLike: dataset if not completely invalid.
        """
        not_finite = ~np.isfinite(X)
        if hasattr(not_finite, "columns"):
            total_sum = not_finite.sum().sum()
        else:
            total_sum = not_finite.sum()
        if total_sum == X.size:
            raise ValueError("Data set contains only missing and/or infinite values.")
        else:
            X = validate_data(self, X, ensure_min_features=2, ensure_all_finite=False)

        return X

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)


def shift_log_transform(values: npt.ArrayLike) -> npt.ArrayLike:
    """Sequential data processing consisting of shifting to avoid negative values
    and subsequent log transformation

    Args:
        values (npt.ArrayLike): raw descriptor values.

    Returns:
        npt.ArrayLike: transformed descriptor values.
    """
    min_val = np.min(values)
    if min_val <= 0:
        shifted = values - min_val + 1e-6
    else:
        shifted = values
    transformed = np.log1p(shifted)
    return transformed


class RDKit2DNovartisScaler(TransformerMixin, BaseEstimator):

    def __init__(self):
        # Retrieve scipy functions
        self.functions = self._get_functions()

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        # Verify is Pandas (does not work without column names)
        self._validate_is_pandas(X)
        # Define n_features and training samples
        self.n_features_in_ = X.shape[1]
        self.n_train_samples_ = X.shape[0]
        self.feature_names_in_ = X.columns.values

        # Store scalers for features without cdf
        self._other_scalers = {}
        for col in X.columns:
            if col not in self.functions.keys():
                self._other_scalers[col] = MinMaxScaler().fit(X[[col]])

        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # Verify array is Pandas Dataframe
        self._validate_is_pandas(X)
        # Run transformation
        features = []
        for col in X.columns:
            if col in self.functions.keys():
                norm = self.functions[col](X[[col]])
                # print("in", norm.shape)
            else:
                norm = self._other_scalers[col].transform(X[[col]])
                # print("out", norm.shape)
            features.append(norm)
        X = np.hstack(features)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            input_features = self.feature_names_in_
        else:
            input_features = np.array(input_features)
        return input_features

    def _get_functions(self):
        cdfs = {}

        for name, (dist, params, minV, maxV, avg, std) in dists.dists.items():
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            if dist in ['gilbrat', 'gibrat']:
                # fix change in scikit learn
                if hasattr(stats, 'gilbrat'):
                    dist = 'gilbrat'
                else:
                    dist = 'gibrat'

            if dist in ['gilbrat', 'gibrat']:
                # fix change in scikit learn
                if hasattr(stats, 'gilbrat'):
                    dist = 'gilbrat'
                else:
                    dist = 'gibrat'

            dist = getattr(stats, dist)

            # make the cdf with the parameters
            def cdf(v, dist=dist, arg=arg, loc=loc, scale=scale, minV=minV, maxV=maxV):
                v = dist.cdf(np.clip(v, minV, maxV), loc=loc, scale=scale, *arg)
                return np.clip(v, 0., 1.)

            cdfs[name] = cdf

        return cdfs

    def _validate_is_pandas(self, X):
        if not hasattr(X, "columns"):
            raise TypeError("Given array is not a Pandas DataFrame")
