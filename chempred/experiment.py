"""
Module containing the Explorer class, which enabling perfoming exploratory experiments.

@author: Dr. Freddy A. Bernal
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, Literal, Optional
# from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.pipeline import Pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm.contrib.itertools import product

from config import CLASSIFIERS, MOL_TRANSFORMERS, SAMPLING_METHODS, SimpleConfig
from preprocessing import RemoveCorrelated, MissingValuesRemover
from utils import add_timing


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
    """Abstract class for exploration. The central method 'evalaute' is defined here."""

    def __init__(
        self,
        ml_algorithms: Union[list, Literal["all"]] = "all",
        mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
        preprocessing: bool = True,  # automatically turned off if fingerprints
        random_state: int = 21,
        n_jobs: int = 1,
        scoring: Optional[list] = None,
    ):
        self.ml_algorithms = ml_algorithms
        self.random_state = random_state
        self.mol_transformers = mol_transformers
        self.preprocessing = preprocessing
        self.n_jobs = n_jobs
        self._data_pipelines = []
        self._steps = []
        # self._set_estimators() TO SET UP IN SUBCLASS
        self._set_scoring_functions(scoring)

    @abstractmethod
    def evaluate(self, X_train, X_test, y_train, y_test):
        # define columns for resulting dataframe
        # define evaluation loop
        pass

    @add_timing
    def _run_evaluation(self, X_train, X_test, y_train, y_test):
        pipe = self._data_pipelines[-1]
        pipe.fit(X_train, y_train)
        scores = self._score_from_predictor(pipe, X_test, y_test)
        return scores

    def _create_pipeline(self, transformer: Optional[list] = None) -> Pipeline:
        if transformer is not None:
            steps = [
                ("SmilesToMolTransformer", SmilesToMolTransformer()),
                ("Standardizer", Standardizer()),
                (transformer[0], transformer[1]()),
            ]
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

    def _instantiate_estimator(self, model):
        params = {}
        if "random_state" in model().get_params().keys():
            params["random_state"] = self.random_state
        if "n_jobs" in model().get_params().keys():
            params["n_jobs"] = self.n_jobs
            return model(**params)
        return model()

    @abstractmethod
    def _score_from_predictor(self, estimator, X, y):
        # any number of scoring functions used. It returns results as an array
        pass

    def _get_steps(self, pipe1, pipe2):
        return list(pipe1.named_steps.items()) + list(pipe2.named_steps.items())

    def _select_best_model(self):
        av = self.results_[self._named_scoring_functions].mean(axis=1)
        self.best_index_ = av.sort_values(ascending=False).index[0]
        steps = self._steps[self.best_index_]
        self.best_estimator_ = Pipeline(steps)

    @property
    def best_score_(self):
        best_score = self.results_.loc[self.best_index_, self._named_scoring_functions]
        return best_score.to_dict()

    def predict(self, X):
        _check_fitted(self)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        _check_fitted(self)
        scores = self._score_from_predictor(self.best_estimator_, X, y)
        return {
            key: float(val)
            for key, val in zip(self._named_scoring_functions, scores)
        }

    @abstractmethod
    def _set_estimators(self):
        # Important to set up CLASSIFIERS or REGRESSORS
        pass

    def _set_custom_estimators(self, custom_methods, all_methods):
        est_list = []
        for method in custom_methods:
            if self._check_implemented_estimator(method, all_methods):
                est_tuple = (method.__name__, method)
                est_list.append(est_tuple)
            else:
                raise NotImplementedError(f"{method=} not implemented.")
        return est_list

    def _check_implemented_estimator(self, estimator, implemented_estimators):
        estimator_classes = [est[1] for est in implemented_estimators]
        if estimator in estimator_classes:
            return True
        return False

    @abstractmethod
    def _set_scoring_functions(self, scoring: Optional[list]):
        if scoring is not None:
            self._named_scoring_functions = scoring
        else:
            pass


class ClassificationExplorer(BaseExplorer):
    def __init__(
            self,
            ml_algorithms: Union[list, Literal["all"]] = "all",
            balancing_samplers: Optional[Union[list, Literal["all"]]] = "all",
            mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
            preprocessing: bool = True,  # automatically turned off if fingerprints
            random_state: int = 21,
            n_jobs: int = 1,
            scoring: Optional[list] = None,
    ):
        super().__init__(
            ml_algorithms=ml_algorithms,
            mol_transformers=mol_transformers,
            preprocessing=preprocessing,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring
        )
        self.balancing_samplers = balancing_samplers
        self._set_estimators()

    def evaluate(self, X_train, X_test, y_train, y_test):
        # define columns for resulting dataframe
        columns = ["Algorithm", "Balancing method"]
        columns.extend(self._named_scoring_functions)
        columns.append("Time")

        if self.mol_transformers is not None:
            columns.insert(2, "Molecular Transformer")
            results = pd.DataFrame(columns=columns)
            for transformer in tqdm(self.mol_transformers,
                                    desc="Featurization:"):
                mol_pipe = self._create_pipeline(transformer)
                X_train_trans = mol_pipe.fit_transform(X_train)
                X_test_trans = mol_pipe.fit_transform(X_test)

                for algorithm, sampler in product(
                    self.ml_algorithms, self.balancing_samplers,
                    desc="Models", position=1, leave=False
                ):
                    self._last_config = SimpleConfig(algorithm, sampler, transformer)
                    # print(self._last_config)
                    if transformer[0] != "MolecularDescriptorTransformer":
                        self.preprocessing = False

                    pipe = self._create_pipeline()
                    self._data_pipelines.append(pipe)
                    self._steps.append(self._get_steps(mol_pipe, pipe))
                    scores = self._run_evaluation(
                        X_train_trans,
                        X_test_trans,
                        y_train,
                        y_test,
                    )
                    res = [algorithm[0], sampler[0], transformer[0]] + scores
                    results.loc[len(results)] = res
        else:
            results = pd.DataFrame(columns=columns)
            for algorithm, sampler in product(
                self.ml_algorithms, self.balancing_samplers,
                desc="Models", position=1, leave=False
            ):
                self._last_config = SimpleConfig(algorithm, sampler)
                pipe = self._create_pipeline()
                self._data_pipelines.append(pipe)
                self._steps.append(pipe)
                scores = self._run_evaluation(X_train, X_test, y_train, y_test)
                res = [algorithm[0], sampler[0]] + scores
                results.loc[len(results)] = res

        self.results_ = results
        self._select_best_model()

        if hasattr(self, "_last_config"):
            delattr(self, "_last_config")

    def _score_from_predictor(self, estimator, X, y):
        y_pred = estimator.predict(X)
        ba = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = self._roc_auc(estimator, X, y)
        return np.array([ba, f1, auc])

    def _roc_auc(self, estimator, X, y):
        try:
            probs = estimator.predict_proba(X)[:, 1]
        except AttributeError:
            probs = estimator.decision_function(X)
        return roc_auc_score(y, probs)

    def _set_scoring_functions(self, scoring):
        if scoring is not None:
            self._named_scoring_functions = scoring
        else:
            self._named_scoring_functions = ["Balanced Accuracy", "F1 score", "ROC AUC"]

    def _set_estimators(self):
        methods = [self.ml_algorithms, self.balancing_samplers, self.mol_transformers]
        names = ["ml_algorithms", "balancing_samplers", "mol_transformers"]
        full_lists = [CLASSIFIERS, SAMPLING_METHODS, MOL_TRANSFORMERS]
        for method, name, full_list in zip(methods, names, full_lists):
            if method == "all":
                setattr(self, name, full_list)
                if name == "balancing_samplers":
                    attr = getattr(self, name)
                    attr.append((None, None))
                    setattr(self, name, attr)
            else:
                if method is None and name == "mol_transformers":
                    pass
                else:
                    custom_list = self._set_custom_estimators(method, full_list)
                    setattr(self, name, custom_list)

    def __str__(self):
        name = type(self).__name__
        attr = self._get_attributes()
        if attr is not None:
            attr_str = "".join([f"{key}={val}, " for key, val in attr.items()])
        else:
            attr_str = ""
        return f"{name}({attr_str})"

    def _get_attributes(self) -> Optional[dict]:
        attr = {}
        if self.ml_algorithms != CLASSIFIERS:
            attr["ml_algorithms"] = [est[1] for est in self.ml_algorithms]
        if self.balancing_samplers != SAMPLING_METHODS:
            attr["balancing_samplers"] = [est[1] for est in self.balancing_samplers]
        if self.mol_transformers != MOL_TRANSFORMERS:
            attr["mol_transformers"] = [est[1] for est in self.mol_transformers]
        if self.preprocessing is False:
            attr["preprocessing"] = self.preprocessing
        if self.random_state != 21:
            attr["random_state"] = self.random_state
        if self.n_jobs != 1:
            attr["n_jobs"] = self.n_jobs
        if self._named_scoring_functions != ["Balanced Accuracy", "F1 score", "ROC AUC"]:
            attr["score"] = self._named_scoring_functions

        return attr if attr else None
