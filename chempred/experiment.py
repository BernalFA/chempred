"""
Module containing the Explorer class, which enabling perfoming exploratory experiments.

@author: Dr. Freddy A. Bernal
"""

from itertools import product

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import CLASSIFIERS, MOL_TRANSFORMERS, SAMPLING_METHODS, SimpleConfig
from preprocessing import RemoveCorrelated
from utils import add_timing


def _check_fitted(cls):
    if not hasattr(cls, "best_index_"):
        raise NotFittedError("No fitted model available. Run evaluate() first.")


class Explorer:
    def __init__(
        self,
        classifiers="all",
        balancing_samplers="all",
        mol_transformers="all",
        preprocessing=True,  # automatically turned off if fingerprints
        random_state=21,
        n_jobs=1,
    ):
        self.random_state = random_state
        self.classifiers = classifiers
        self.balancing_samplers = balancing_samplers
        self.mol_transformers = mol_transformers
        self.preprocessing = preprocessing
        self.n_jobs = n_jobs
        self._data_pipelines = []
        self._steps = []
        self._set_estimators()

    def evaluate(self, X_train, X_test, y_train, y_test):
        columns = [
            "Classifier",
            "Balancing method",
            "Balanced Accuracy",
            "F1 score",
            "ROC AUC",
            "Time",
        ]
        if self.mol_transformers is not None:
            columns.insert(2, "Molecular Transformer")
            results = pd.DataFrame(columns=columns)
            for transformer in self.mol_transformers:
                mol_pipe = self._create_pipeline(transformer)
                X_train = mol_pipe.fit_transform(X_train)
                X_test = mol_pipe.fit_transform(X_test)

                for classifier, sampler in product(
                    self.classifiers, self.balancing_samplers
                ):
                    self._last_config = SimpleConfig(classifier, sampler, transformer)
                    if transformer[0] != "MolecularDescriptorTransformer":
                        self.preprocessing = False

                    pipe = self._create_pipeline()
                    self._data_pipelines.append(pipe)
                    self._steps.append(self._get_steps(mol_pipe, pipe))
                    scores = self._run_evaluation(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                    )
                    res = [classifier[0], sampler[0], transformer[0]] + scores
                    results.loc[len(results)] = res
        else:
            results = pd.DataFrame(columns=columns)
            for classifier, sampler in product(
                self.classifiers, self.balancing_samplers
            ):
                self._last_config = SimpleConfig(classifier, sampler)
                pipe = self._create_pipeline()
                self._data_pipelines.append(pipe)
                self._steps.append(pipe)
                scores = self._run_evaluation(X_train, X_test, y_train, y_test)
                res = [classifier[0], sampler[0]] + scores
                results.loc[len(results)] = res

        self.results_ = results
        self._select_best_model()

        if hasattr(self, "_last_config"):
            delattr(self, "_last_config")

    @add_timing
    def _run_evaluation(self, X_train, X_test, y_train, y_test):
        pipe = self._data_pipelines[-1]
        pipe.fit(X_train, y_train)
        scores = self._score_from_predictor(pipe, X_test, y_test)
        return scores

    def _create_pipeline(self, transformer=None):
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

    @property
    def best_score_(self):
        cols = ["Balanced Accuracy", "F1 score", "ROC AUC"]
        best_score = self.results_.loc[self.best_index_, cols]
        return best_score.to_dict()

    def predict(self, X):
        _check_fitted(self)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        _check_fitted(self)
        cols = ["Balanced Accuracy", "F1 score", "ROC AUC"]
        scores = self._score_from_predictor(self.best_estimator_, X, y)
        return {key: float(val) for key, val in zip(cols, scores)}

    def _select_best_model(self):
        cols = ["Balanced Accuracy", "F1 score", "ROC AUC"]
        av = self.results_[cols].mean(axis=1)
        self.best_index_ = av.sort_values(ascending=False).index[0]
        steps = self._steps[self.best_index_]
        self.best_estimator_ = Pipeline(steps)

    def _check_implemented_estimator(self, estimator, implemented_estimators):
        estimator_classes = [est[1] for est in implemented_estimators]
        if estimator in estimator_classes:
            return True
        return False

    def _get_steps(self, pipe1, pipe2):
        return list(pipe1.named_steps.items()) + list(pipe2.named_steps.items())

    def _set_custom_estimators(self, custom_methods, all_methods):
        est_list = []
        for method in custom_methods:
            if self._check_implemented_estimator(method, all_methods):
                est_tuple = (method.__name__, method)
                est_list.append(est_tuple)
            else:
                raise NotImplementedError(f"{method=} not implemented.")
        return est_list

    def _set_estimators(self):
        methods = [self.classifiers, self.balancing_samplers, self.mol_transformers]
        names = ["classifiers", "balancing_samplers", "mol_transformers"]
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
