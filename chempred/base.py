"""
Module containing the BaseExplorer class necessary to create custom explorers.

@author: Dr. Freddy A. Bernal
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, Literal, Optional

import numpy as np
import numpy.typing as npt
from imblearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from chempred.config import ExplorerConfig
from chempred.utils import add_timing


Preprocessing = Optional[Literal["StandardScaler", "NovartisScaler", "NoScaler"]]


def _check_fitted(cls: Callable):
    """Simple helper to check the Explorer is fitted before calling predict or score

    Args:
        cls (Callable): class derived from BaseExplorer

    Raises:
        NotFittedError: if Explorer has not run yet evaluate() method when calling
                        predict or score on additional data.
    """

    if not hasattr(cls, "best_index_"):
        raise NotFittedError("No fitted model available. Run evaluate() first.")


class BaseExplorer(ABC):
    """Abstract class for exploration. The central method `evaluate` is defined here."""

    def __init__(
        self,
        ml_algorithms: Union[list, Literal["all"]] = "all",
        mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
        preprocessing: Preprocessing = None,
        random_state: int = 21,
        n_jobs: int = 1,
        scoring: Optional[list] = None,
        select_best_by: str = "average",
    ):
        """
        Args:
            ml_algorithms (list | 'all', optional): ML algorithms to include in
                    exploration. Defaults to "all" (include all the implemented models).
            mol_transformers (list | 'all' | None, optional): molecular transformers
                    to include in exploration. Defaults to "all" (include all the
                    implemented transformers).
            preprocessing ("StandardScaler" | "NovartisScaler" | "NoScaler" | None,
                    optional): data preprocessing applied before training the model.
                    Defaults to `None`.
                    If preprocessing is not None, features with missing values,
                    highly correlated, or low variance will be removed. If molecular
                    transformation to fingerprints, `preprocessing` will be ignored.
                    "NovartisScaler" refers to `RDKit2DNovartisScaler` in the
                    preprocessing module. The scaler is based on Cumulative Distribution
                    Functions (CDFs) defined by Novartis some years ago and made
                    available in the `descriptastorus` package. There are CDFs for 200
                    descriptors (out of 217 in RDKit 2025). Thus, descriptors without
                    CDF in `descriptastorus` are scaled using MinMaxScaler.
                    "StandardScaler" refers to standard scaler from sklearn.
                    "NoScaler" means no scaling is performed.
            n_jobs (int, optional): number of cpu units for pipeline processing (used
                    on algorithms that allows multiprocessing). Defaults to 1.
            scoring (list | None, optional): names given to the scoring functions
                    used during evaluation. Defaults to None assign scoring as balanced
                    accuracy.
            select_best_by (str | list): mode of selection of best performing pipeline.
                    The name of a particular metrics used in `scoring` can be used.
                    Defaults to 'average', indicating that all the calculated metrics
                    will be averaged and the highest average value will be used to
                    define the best model. If a list is given, those metrics will be
                    averaged.
        """

        self.params = ExplorerConfig(
            ml_algorithms=ml_algorithms,
            balancing_samplers=None,
            mol_transformers=mol_transformers,
            preprocessing=preprocessing,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring,
            select_best_by=select_best_by
        )
        self._data_pipelines = []
        self._steps = []
        # self._set_estimators() TO SET UP IN SUBCLASS
        self.scorers = self._set_scoring_functions(scoring)
        self._select_best_by = self._check_metrics_for_selection(select_best_by)
        self._from_descriptors = False

    @abstractmethod
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Central method for automatic evaluation of multiple pipelines"""
        # define columns for resulting dataframe
        # define evaluation loop with _run_evaluation
        # store results as attribute results_
        # finally _select_best_model
        pass

    @abstractmethod
    def _set_estimators(self):
        """Check provided estimators or assign 'all' estimators available"""
        # Important to set up CLASSIFIERS or REGRESSORS
        # use _set_custom_estimators
        # this method needs to be run at initialiation.
        pass

    @abstractmethod
    def _set_scoring_functions(self, scoring: Optional[list]):
        """Help define the scoring functions used during model evaluation"""
        pass

    @property
    def best_score_(self) -> dict:
        """Locate the scores of the best pipeline (using best_index_ as defined in
        _select_best_model)

        Returns:
            dict: test scores obtained for the best pipeline
        """
        cols = [scorer[0] for scorer in self.scorers]
        best_score = self.results_.loc[self.best_index_, cols]
        return best_score.to_dict()

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict label/target value for given dataset X using the best pipeline from
        the evaluate() method.

        Args:
            X (npt.ArrayLike): dataset for prediction (smiles or features).

        Returns:
            np.ndarray: predicted target values / labels
        """
        _check_fitted(self)
        return self.best_estimator_.predict(X)

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike) -> dict:
        """Evaluate the prediction performance of the best pipeline on the target
        value (y) for the given dataset (X).

        Args:
            X (npt.ArrayLike): dataset for prediction (smiles or features).
            y (npt.ArrayLike): target values / labels.

        Returns:
            dict: set of scores to assess performance of predictions.
        """
        _check_fitted(self)
        scores = self._score_from_predictor(self.best_estimator_, X, y)
        cols = [scorer[0] for scorer in self.scorers]
        return {key: float(val) for key, val in zip(cols, scores)}

    @add_timing
    def _run_evaluation(
        self,
        X_train: npt.ArrayLike,
        X_test: npt.ArrayLike,
        y_train: npt.ArrayLike,
        y_test: npt.ArrayLike,
    ) -> np.ndarray:
        """Fit pipeline on training data and calculate performance on test data.

        Args:
            X_train (npt.ArrayLike): training data or smiles
            X_test (npt.ArrayLike): test data or smiles
            y_train (npt.ArrayLike): training labels/target
            y_test (npt.ArrayLike): test labels/target

        Returns:
            np.ndarray: performance scores on test data
        """
        pipe = self._data_pipelines[-1]
        try:
            pipe.fit(X_train, y_train)
            scores = self._score_from_predictor(pipe, X_test, y_test)
        except ValueError:
            scores = {scorer[0]: np.nan for scorer in self.scorers}
        return scores

    def _score_from_predictor(
        self, estimator: Pipeline, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> dict:
        """Assess performance of given estimator on the provided dataset using selected
        scoring metrics.

        Args:
            estimator (Pipeline): pipeline containing an ML model
            X (npt.ArrayLike): features
            y (npt.ArrayLike): labels

        Returns:
            dict: performance scores
        """
        y_pred = estimator.predict(X)

        if any(scorer[0] in ["roc_auc", "prc_auc"] for scorer in self.scorers):
            try:
                probs = estimator.predict_proba(X)[:, 1]
            except AttributeError:
                probs = estimator.decision_function(X)

        calc_scores = {}
        for scorer in self.scorers:
            if scorer[0] not in ["roc_auc", "prc_auc"]:
                value = scorer[1](y, y_pred)
            else:
                value = scorer[1](y, probs)
            calc_scores[scorer[0]] = value

        return calc_scores

    def _select_best_pipeline(self):
        """Define best model from obtained performance metrics. Results are stored as
        attributes best_index_ and best_estimator_
        """
        scorers = [scorer[0] for scorer in self.scorers]
        if isinstance(self._select_best_by, str) and self._select_best_by != "average":
            sorting_df = self.results_[self._select_best_by].copy()
        else:
            results = self.results_.copy()
            cols_selection = []
            for name in scorers:
                if name in ["mcc", "cohen_kappa"]:
                    results["n_" + name] = (results[name] + 1) / 2
                    cols_selection.append("n_" + name)
                elif name == "r2":
                    results["1-" + name] = 1 - results[name]
                    cols_selection.append("1-" + name)
                else:
                    cols_selection.append(name)
            sorting_df = results[cols_selection].mean(axis=1)

        self.best_index_ = sorting_df.sort_values(ascending=False).index[0]
        steps = self._steps[self.best_index_]
        self.best_estimator_ = Pipeline(steps)

    def _set_custom_estimators(self, custom_methods: list, all_methods: list) -> list:
        """Iterate over custom_methods to assess whether the given estimator/method is
        implemented.

        Args:
            custom_methods (list): method to use in exploration
                                   (e.g. [RandomForestClassifier])
            all_methods (list): available estimators as defined in config.py
                                (e.g. CLASSIFIERS)

        Raises:
            NotImplementedError: raise an error if any of the provided custom_methods
                                 is not yet implemented in the Explorer.

        Returns:
            list: given estimators as tuples (name, estimator)
        """
        est_list = []
        for method in custom_methods:
            if self._check_implemented_estimator(method, all_methods):
                est_tuple = (method.__name__, method)
                est_list.append(est_tuple)
            else:
                raise NotImplementedError(f"{method=} not implemented.")
        return est_list

    @staticmethod
    def _check_implemented_estimator(estimator, implemented_estimators):
        """Verify estimator is included within implemented estimators in config.py

        Returns:
            bool: True if estimator is implemented
        """
        estimator_classes = [est[1] for est in implemented_estimators]
        if estimator in estimator_classes:
            return True
        return False

    def _check_metrics_for_selection(self, metrics: Union[list, str]) -> list:
        """Check for correctness the given method for selection of the best pipeline.

        Args:
            metrics (list | str): evaluation metrics used for selection of best
                                  pipeline. If a list of metrics is given, their
                                  average will be used to select the best pipeline.
                                  Defaults to 'average' on all the metrics used
                                  during evaluation.

        Raises:
            ValueError: raise error if given metrics not present in the set of
                        evaluation metrics.

        Returns:
            list: metrics
        """
        scorers = [scorer[0] for scorer in self.scorers]
        if isinstance(metrics, str):
            if metrics in scorers + ["average"]:
                return metrics
        elif isinstance(metrics, list):
            if set(metrics).issubset(scorers):
                return metrics
            else:
                raise ValueError(
                    f"{metrics} not in agreement with selected scoring functions."
                )
        else:
            raise ValueError(
                f"{metrics} not in agreement with selected scoring functions."
            )

    def _get_steps(self, pipe1: Pipeline, pipe2: Pipeline) -> list[tuple]:
        """Unify steps of the pipelines for molecular transformation and data processing
        and training.

        Args:
            pipe1 (Pipeline): molecular transformation pipeline.
            pipe2 (Pipeline): ML training pipeline.

        Returns:
            list[tuple]: full sequence of steps followed.
        """
        return list(pipe1.named_steps.items()) + list(pipe2.named_steps.items())
