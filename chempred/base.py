"""
Module containing the Explorer class, which enables perfoming exploratory experiments.

@author: Dr. Freddy A. Bernal
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, Literal, Optional

import numpy as np
import numpy.typing as npt
from imblearn.pipeline import Pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from chempred.preprocessing import RemoveCorrelated, MissingValuesRemover
from chempred.utils import add_timing


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
    """Abstract class for exploration. The central method 'evaluate' is defined here."""

    def __init__(
        self,
        ml_algorithms: Union[list, Literal["all"]] = "all",
        mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
        preprocessing: bool = True,  # automatically turned off if fingerprints
        random_state: int = 21,
        n_jobs: int = 1,
        scoring: Optional[list] = None,
    ):
        """
        Args:
            ml_algorithms (list | 'all', optional): ML algorithms to include in
                    exploration. Defaults to "all" (include all the implemented models).
            mol_transformers (list | 'all' | None, optional): molecular transformers
                    to include in exploration. Defaults to "all" (include all the
                    implemented transformers).
            preprocessing (bool, optional): if True, a data preprocessing pipeline will
                    be applied before training the model. Defaults to True. Only
                    applicable for molecular descriptors.
            n_jobs (int, optional): number of cpu units for pipeline processing (used
                    on algorithms that allows multiprocessing). Defaults to 1.
            scoring (list | None, optional): names given to the scoring functions
                    used during evaluation. Defaults to None assign scoring as balanced
                    accuracy.
        """

        self.ml_algorithms = ml_algorithms
        self.random_state = random_state
        self.mol_transformers = mol_transformers
        self.preprocessing = preprocessing
        self.n_jobs = n_jobs
        self._data_pipelines = []
        self._steps = []
        # self._set_estimators() TO SET UP IN SUBCLASS
        self.scorers = self._set_scoring_functions(scoring)

    @abstractmethod
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Central method for automatic evaluation of multiple pipelines"""
        # define columns for resulting dataframe
        # define evaluation loop with _run_evaluation
        # store results as attribute results_
        # finally _select_best_model
        pass

    @abstractmethod
    def _score_from_predictor(self, estimator, X, y):
        """Required implementation for performance assessment"""
        # any number of scoring functions used. It returns results as an array
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

    @add_timing
    def _run_evaluation(
        self,
        X_train: npt.ArrayLike,
        X_test: npt.ArrayLike,
        y_train: npt.ArrayLike,
        y_test: npt.ArrayLike
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
        pipe.fit(X_train, y_train)
        scores = self._score_from_predictor(pipe, X_test, y_test)
        return scores

    def _create_pipeline(self, transformer: Optional[list] = None) -> Pipeline:
        """Systematically create a transformation pipeline or a data processing and ML
        training pipeline.

        Args:
            transformer (list | None, optional): molecular transformer to use after
                    smiles to mol transformation and standardization. If given, it will
                    exclusively create a pipeline for molecular transformation.
                    Defaults to None to consider data preprocessing and modeling only.

        Returns:
            Pipeline: instantiated imblearn/sklearn pipeline
        """
        # if molecular transformer given, create transformer pipeline
        if transformer is not None:
            steps = [
                ("SmilesToMolTransformer", SmilesToMolTransformer()),
                ("Standardizer", Standardizer()),
                (transformer[0], transformer[1]()),
            ]
        # if not molecular transformer, create ML pipeline
        else:
            steps = [
                (
                    self._last_config.classifier[0],
                    self._instantiate_estimator(self._last_config.classifier[1]),
                )
            ]
            if self._last_config.sampler[1] is not None:
                steps.insert(
                    0,
                    (
                        self._last_config.sampler[0],
                        self._instantiate_estimator(self._last_config.sampler[1]),
                    ),
                )
            if self.preprocessing:
                preprocess = [
                    ("MissingValuesRemover", MissingValuesRemover()),
                    ("RemoveCorrelated", RemoveCorrelated()),
                    ("VarianceThreshold", VarianceThreshold()),
                    ("StandardScaler", StandardScaler()),
                ]
                steps = preprocess + steps

        return Pipeline(steps=steps)

    def _instantiate_estimator(self, estimator):
        """Instantiate given estimator after setting up 'random_state' and 'n_jobs' if
        possible (depending on whether the implementation uses them).

        Args:
            estimator: sklearn, imblearn or scikit-mol estimator

        Returns:
            instantiated estimator
        """
        from lightgbm import LGBMClassifier
        params = {}
        if "random_state" in estimator().get_params().keys():
            params["random_state"] = self.random_state
        if "n_jobs" in estimator().get_params().keys():
            params["n_jobs"] = self.n_jobs
        if estimator == LGBMClassifier:
            params["verbose"] = -1
            return estimator(**params)
        return estimator()

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

    @abstractmethod
    def _select_best_pipeline(self):
        """Define best model from performance metrics. Results are
        stored as the attributes best_index_ and best_estimator_
        """
        pass

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
        return {
            key: float(val)
            for key, val in zip(cols, scores)
        }
