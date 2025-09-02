"""
Sklearn compatible transformers for removal of correlated features up to defined
threshold and removal of missing values.

@author: Dr. Freddy A. Bernal
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils.validation import check_is_fitted, validate_data
from scipy import stats


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

    Example:
        ```python
        remover = MissingValuesRemover()
        X_processed = remover.fit_transform(X)
        ```
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


def shift_log_transform(values):
    min_val = np.min(values)
    if min_val <= 0:
        shifted = values - min_val + 1e-6
    else:
        shifted = values
    transformed = np.log1p(shifted)
    return transformed


# Pipeline to transform descriptors on counts or discrete small integers
count_based_transform_pipeline = Pipeline([
    ("ShiftLogTransformer", FunctionTransformer(shift_log_transform, validate=True)),
    ("MinMaxScaler", MinMaxScaler())
])


# Pipeline to transform Heavily-tailed continuous values
heavy_skew_transform_pipeline = Pipeline([
    ("ShiftLogTransformer", FunctionTransformer(shift_log_transform, validate=True)),
    ("StandardScaler", StandardScaler())
])


class RDKit2DScaler(TransformerMixin, BaseEstimator):
    def __init__(self, skewness_threshold):
        super().__init__()
        self.skewness_threshold = skewness_threshold

    def fit(self, X, y=None):
        # Define n_features and training samples
        self.n_features_in_ = X.shape[1]
        self.n_train_samples_ = X.shape[0]
        # Define groups
        mask = self._get_transformation_mask(X)
        # Create transformer
        self.transformer = self._create_transformer(mask)
        if self.transformer is not None:
            self.transformer.fit(X=X, y=y)
        else:
            raise ValueError("Features could not be transformed")
        return self

    def transform(self, X):
        # Check fitted as used by sklearn e.g. in VarianceThreshold class
        check_is_fitted(self)
        # Run transformation
        X = self.transformer.transform(X)
        return X

    def _get_transformation_mask(self, X):
        # Define groups and empty dict for indices
        groups = ["binary", "count", "bounded", "heavy_skew", "moderate_skew"]
        mask = {group: [] for group in groups}
        # Iterate over descriptors
        for i in range(X.shape[1]):
            single_desc = X[:, i]
            # get descriptiive info
            unique_vals = np.unique(single_desc)
            min_val, max_val = np.min(single_desc), np.max(single_desc)
            n_unique = len(unique_vals)

            # Case 1: binary descriptor
            if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
                mask["binary"].append(i)
                continue

            # Case 2: counts or discrete small integers
            if max_val <= 20:
                mask["count"].append(i)
                continue

            # Case 3: bounded continuous [0, 1]
            if min_val >= 0 and max_val <= 1:
                mask["bounded"].append(i)
                continue

            # Case 4: Heavy-tailed continuous (skew > threshold)
            skew_val = stats.skew(single_desc)
            if skew_val > self.skewness_threshold:
                mask["heavy_skew"].append(i)
                continue

            # Case 5: continuous with moderate skew
            mask["moderate_skew"].append(i)

        return mask

    def _create_transformer(self, mask):
        processes = []
        if mask["count"]:
            processes.append((
                "CountTransformer", count_based_transform_pipeline, mask["count"]
            ))
        if mask["heavy_skew"]:
            processes.append((
                "HeavySkewTransformer",
                heavy_skew_transform_pipeline,
                mask["heavy_skew"]
            ))
        if mask["moderate_skew"]:
            processes.append((
                "ModerateSkewTransformer", StandardScaler(), mask["moderate_skew"]
            ))
        other = mask["binary"] + mask["bounded"]
        if other:
            # do nothing
            processes.append((
                'passthrough_cols', 'passthrough', other
            ))
        return ColumnTransformer(processes)
