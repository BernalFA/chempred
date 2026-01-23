"""
Module containing the Explorer classes, which enable perfoming exploratory experiments.

@author: Dr. Freddy A. Bernal
"""

from typing import Union, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.pipeline import Pipeline

try:
    from IPython import get_ipython

    if get_ipython():
        if "IPKernelApp" in get_ipython().config:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm
from tqdm.contrib.itertools import product

from chempred.base import BaseExplorer
from chempred.config import (
    CLASSIFIERS, REGRESSORS, MOL_TRANSFORMERS, SAMPLING_METHODS, SimpleConfig, SCORING
)
from chempred.pipeline import create_pipeline


class ClassificationExplorer(BaseExplorer):
    """Facilitate training and performance assessment of a series of automatically
    created pipelines, combining molecular transformations implemented in `rdkit` and
    provided in `scikit-mol`, class imbalance sampling techniques implemented in
    `imblearn`, and machine learning methods implemented in `sklearn`. Each pipeline is
    evaluated for predictive performance on a separate dataset (validation set) using
    different available metrics commonly used in classification problems (e.g. balanced
    accuracy, F1 score, ROC AUC). Afterward, the best performing pipeline (according to
    a single metric or an average of several) is made available for prediction of labels
    on unseen molecular structures. Scoring on specified datasets is also possible
    (further evaluation).

    The current implementation uses default values for all the parameters in the
    considered estimators, except for `random_state` and `n_jobs`, which can be
    configured upon instance definition.
    """

    def __init__(
        self,
        ml_algorithms: Union[list, Literal["all"]] = "all",
        balancing_samplers: Optional[Union[list, Literal["all"]]] = "all",
        mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
        preprocessing: Optional[Literal["StandardScaler", "RDKit2DScaler"]] = None,
        random_state: int = 21,
        n_jobs: int = 1,
        scoring: Optional[list] = None,
        select_best_by: str = "average",
    ):
        """
        Args:
            ml_algorithms (list | "all", optional): ML algorithms to include in
                    exploration. Defaults to "all" (include all the implemented models).
            balancing_samplers (list | 'all' | None, optional): data samplers for class
                    imbalance treatment. Defaults to "all" (include all the implemented
                    samplers).
            mol_transformers (list | 'all' | None, optional): molecular transformers
                    to include in exploration. Defaults to "all" (include all the
                    implemented transformers).
            preprocessing ("StandardScaler" | "NovartisScaler" | "NoScaler" | None,
                    optional): data preprocessing applied before training the model.
                    Defaults to `None`.
                    If preprocessing is not `None`, features with missing values,
                    highly correlated, or habing low variance will be removed. If
                    molecular transformation is set to fingerprints, `preprocessing`
                    will be ignored.
                    "NovartisScaler" refers to `RDKit2DNovartisScaler` in the
                    preprocessing module. The scaler is based on Cumulative Distribution
                    Functions (CDFs) defined by Novartis some years ago and made
                    available in the `descriptastorus` package. There are CDFs for 200
                    descriptors (out of 217 in RDKit 2025). Thus, descriptors without
                    CDF in `descriptastorus` are scaled using MinMaxScaler.
                    "StandardScaler" refers to standard scaler from sklearn.
                    "NoScaler" means no scaling is performed.
            n_jobs (int, optional): number of cpu cores for pipeline processing (used
                    on algorithms that allows multiprocessing). Defaults to 1.
            scoring (list | None, optional): names given to the scoring functions
                    used during evaluation. Defaults to `None` to assign default names.
            select_best_by (str | list): mode of selection of best performing pipeline.
                    The name of a particular metrics used in `scoring` can be used.
                    Defaults to 'average', indicating that all the calculated metrics
                    will be averaged and the highest average value will be used to
                    define the best model. If a list is given, those metrics will be
                    averaged.
        """
        super().__init__(
            ml_algorithms=ml_algorithms,
            mol_transformers=mol_transformers,
            preprocessing=preprocessing,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring,
        )
        self.balancing_samplers = balancing_samplers
        self._select_best_by = self._check_metrics_for_selection(select_best_by)
        self._set_estimators()

    def __str__(self):
        name = type(self).__name__
        attr = self._get_non_default_params()
        if attr is not None:
            attr_str = "".join([f"{key}={val}, " for key, val in attr.items()])
        else:
            attr_str = ""
        return f"{name}({attr_str})"

    def evaluate(
        self,
        X_train: npt.ArrayLike,
        X_test: npt.ArrayLike,
        y_train: npt.ArrayLike,
        y_test: npt.ArrayLike,
    ):
        """Sequentially create and train pipelines with the training dataset, and assess
        performance using the test dataset. During execution, the following private
        attributes are set:
        - `_data_pipelines`: list containing all the created pipelines for data
                             preprocessing and ML modeling.
        - `_steps`: list containing the steps for the full pipeline including molecular
                    transformations.
        After evaluation is complete, the following attributes are set:
        - `results_`: pd.DataFrame summarizing the results from all the evaluated
                      pipelines.
        - `best_estimator_`: best pipeline chosen according to 'select_best_by'.
        - `best_index_`: index of the best pipeline (in `results_`).

        Args:
            X_train (npt.ArrayLike): training smiles / features
            X_test (npt.ArrayLike): test smiles / features
            y_train (npt.ArrayLike): training labels
            y_test (npt.ArrayLike): test labels
        """
        results = []
        # Run iterative training and evaluation
        if self.mol_transformers is not None:
            # columns.insert(2, "Molecular Transformer")
            for transformer in tqdm(self.mol_transformers, desc="Overall progress"):
                config = SimpleConfig((None, None), transformer=transformer)
                mol_pipe = create_pipeline(config=config,
                                           preprocessing=None,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs,
                                           mol_only=True)
                X_train_trans = mol_pipe.fit_transform(X_train)
                X_test_trans = mol_pipe.transform(X_test)

                for algorithm, sampler in product(
                    self.ml_algorithms,
                    self.balancing_samplers,
                    desc=f"Models with {transformer[0].replace("Transformer", "")}",
                    position=1,
                    leave=False,
                ):
                    if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                        continue
                    if transformer[0] != "MolecularDescriptorTransformer":
                        self._from_descriptors = False
                    else:
                        self._from_descriptors = True

                    config = SimpleConfig(algorithm, sampler)
                    pipe = create_pipeline(config=config,
                                           preprocessing=self.preprocessing,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs)
                    self._data_pipelines.append(pipe)
                    self._steps.append(self._get_steps(mol_pipe, pipe))
                    scores = self._run_evaluation(
                        X_train_trans,
                        X_test_trans,
                        y_train,
                        y_test,
                    )
                    res = {
                        "Algorithm": algorithm[0],
                        "Balancing method": sampler[0],
                        "Molecular Transformer": transformer[0]
                    }
                    res.update(scores)
                    results.append(res)
        else:
            for algorithm, sampler in product(
                self.ml_algorithms, self.balancing_samplers, desc="Models", position=1,
                leave=False
            ):
                if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                    continue
                config = SimpleConfig(algorithm, sampler)
                pipe = create_pipeline(config=config,
                                       preprocessing=None,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)
                self._data_pipelines.append(pipe)
                self._steps.append(pipe)
                scores = self._run_evaluation(X_train, X_test, y_train, y_test)
                res = {
                    "Algorithm": algorithm[0],
                    "Molecular Transformer": transformer[0]
                }
                res.update(scores)
                results.append(res)

        # store results and select best performing pipeline
        self.results_ = pd.DataFrame(results)
        self._select_best_pipeline()

    def _score_from_predictor(
        self, estimator: Pipeline, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> np.ndarray:
        """Assess performance of a given estimator on the provided dataset using chosen
        scoring metrics.

        Args:
            estimator (Pipeline): pipeline containing an ML model
            X (npt.ArrayLike): features
            y (npt.ArrayLike): labels

        Returns:
            np.ndarray: performance scores
        """
        y_pred = estimator.predict(X)
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

    def _set_estimators(self):
        """Help to set all estimators from user input (attributes ml_algorithms,
        balancing_samplers, and mol_transformers set using required format).
        """
        methods = [self.ml_algorithms, self.balancing_samplers, self.mol_transformers]
        names = ["ml_algorithms", "balancing_samplers", "mol_transformers"]
        full_lists = [CLASSIFIERS, SAMPLING_METHODS.copy(), MOL_TRANSFORMERS]
        for method, name, full_list in zip(methods, names, full_lists):
            if method == "all":
                setattr(self, name, full_list)
                if name == "balancing_samplers":
                    attr = getattr(self, name)
                    attr.append((None, None))
                    setattr(self, name, attr)
            elif method is None and name == "mol_transformers":
                pass
            elif method is None and name == "balancing_samplers":
                setattr(self, name, [(None, None)])
            else:
                custom_list = self._set_custom_estimators(method, full_list)
                setattr(self, name, custom_list)

    def _select_best_pipeline(self):
        """Define best model from obtained performance metrics. Results are stored as
        attributes `best_index_` and `best_estimator_`
        """
        exclude = ["mcc", "cohen_kappa"]
        scorers = [scorer[0] for scorer in self.scorers]
        if isinstance(self._select_best_by, str) and self._select_best_by != "average":
            sorting_df = self.results_[self._select_best_by].copy()
        else:
            results = self.results_.copy()
            cols_selection = []
            for name in scorers:
                if name in exclude:
                    results["n_" + name] = (results[name] + 1) / 2
                    cols_selection.append("n_" + name)
                else:
                    cols_selection.append(name)
            sorting_df = results[cols_selection].mean(axis=1)

        self.best_index_ = sorting_df.sort_values(ascending=False).index[0]
        steps = self._steps[self.best_index_]
        self.best_estimator_ = Pipeline(steps)

    def _get_non_default_params(self) -> Optional[dict]:
        """Help to get custom attributes on defined instance to use in __str__

        Returns:
            Optional[dict]: attributes as key:value pairs if custom attributes. None is
            returned when all attributes are set as default.
        """
        attr = {}
        if self.ml_algorithms != CLASSIFIERS:
            attr["ml_algorithms"] = [est[1] for est in self.ml_algorithms]
        if self.balancing_samplers != SAMPLING_METHODS:
            attr["balancing_samplers"] = [est[1] for est in self.balancing_samplers]
        if self.mol_transformers != MOL_TRANSFORMERS:
            attr["mol_transformers"] = [est[1] for est in self.mol_transformers]
        if self.preprocessing is not None:
            attr["preprocessing"] = self.preprocessing
        if self.random_state != 21:
            attr["random_state"] = self.random_state
        if self.n_jobs != 1:
            attr["n_jobs"] = self.n_jobs
        if self.scorers != [(
            "balanced_accuracy", SCORING.classification["balanced_accuracy"]
        )]:
            attr["scoring"] = [scorer[0] for scorer in self.scorers]

        return attr if attr else None

    def _set_scoring_functions(self, scoring: Optional[list]) -> list:
        """Help define the scoring functions used during model evaluation.

        Args:
            scoring (Optional[list]): names of scoring function to be used.
                                      If None is provided, balanced_accuracy will
                                      be used.

        Raises:
            ValueError: raise error if 'scoring' is not list or None.

        Returns:
            list: scorers (sklearn or custom scoring functions) used for performance
            evaluation.
        """
        if isinstance(scoring, list):
            scorers = []
            for scorer in scoring:
                try:
                    func = SCORING.classification[scorer]
                    scorers.append((scorer, func))
                except KeyError as err:
                    err.add_note(
                        f"{scorer=} not recognized. Please check available scorers:"
                    )
                    err.add_note("from chempred.config import get_scorer_names")
                    err.add_note("print(get_scorer_names())")
                    raise
        elif scoring is None:
            scorers = [(
                "balanced_accuracy", SCORING.classification["balanced_accuracy"]
            )]
        else:
            raise ValueError("'scoring' accept as inputs lists or None")
        return scorers


class RegressionExplorer(BaseExplorer):
    """Facilitate training and performance assessment of a series of automatically
    created pipelines, combining molecular transformations implemented in `rdkit` and
    provided in `scikit-mol`, and machine learning methods implemented in `sklearn`.
    Each pipeline is evaluated for predictive performance on a separate dataset
    (validation set) by a user-defined evaluation metrics. Afterward, the best
    performing pipeline (by a specific metric or the average of several) is made
    available for predictions on unseen molecular structures. Scoring on specified
    datasets is also possible (further evaluation).

    The current implementation uses default values for all the parameter of the
    considered estimators, except for `random_state` and `n_jobs`, which can be
    configured upon instance definition.
    """

    def __init__(
        self,
        ml_algorithms: Union[list, Literal["all"]] = "all",
        mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
        preprocessing: Optional[Literal["StandardScaler", "RDKit2DScaler"]] = None,
        random_state: int = 21,
        n_jobs: int = 1,
        scoring: Optional[list] = None,
        select_best_by: str = "average",
    ):
        """
        Args:
            ml_algorithms (list | "all", optional): ML algorithms to include in
                    exploration. Defaults to "all" (include all the implemented models).
            mol_transformers (list | 'all' | None, optional): molecular transformers
                    to include in exploration. Defaults to "all" (include all the
                    implemented transformers).
            preprocessing ("StandardScaler" | "NovartisScaler" | "NoScaler" | None,
                    optional): data preprocessing applied before training the model.
                    Defaults to `None`.
                    If preprocessing is not `None`, features with missing values,
                    highly correlated, or having low variance will be removed. If
                    molecular transformation set to fingerprints, `preprocessing` will
                    be ignored.
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
                    used during evaluation. Defaults to None to assign default names.
            select_best_by (str | list): mode of selection of best performing pipeline.
                    The name of a particular metrics used in `scoring` can be used.
                    Defaults to 'average', indicating that all the calculated metrics
                    will be averaged and the highest average value will be used to
                    define the best model. If a list is given, those metrics will be
                    averaged.
        """
        super().__init__(
            ml_algorithms=ml_algorithms,
            mol_transformers=mol_transformers,
            preprocessing=preprocessing,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring,
        )
        self._select_best_by = self._check_metrics_for_selection(select_best_by)
        self._set_estimators()

    def __str__(self):
        name = type(self).__name__
        attr = self._get_non_default_params()
        if attr is not None:
            attr_str = "".join([f"{key}={val}, " for key, val in attr.items()])
        else:
            attr_str = ""
        return f"{name}({attr_str})"

    def evaluate(
        self,
        X_train: npt.ArrayLike,
        X_test: npt.ArrayLike,
        y_train: npt.ArrayLike,
        y_test: npt.ArrayLike,
    ):
        """Sequentially create and train pipelines with the training dataset, and assess
        performance using the test dataset. During execution, the following private
        attributes are set:
        - `_data_pipelines`: list containing all the created pipelines for data
                             preprocessing and ML modeling.
        - `_steps`: list containing the steps for the full pipeline including molecular
                    transformations.
        After evaluation is complete, the following attributes are set:
        - `results_`: pd.DataFrame summarizing the results from all the evaluated
                      pipelines.
        - `best_estimator_`: best pipeline chosen according to 'select_best_by'.
        - `best_index_`: index of the best pipeline (in `results_`).

        Args:
            X_train (npt.ArrayLike): training smiles / features
            X_test (npt.ArrayLike): test smiles / features
            y_train (npt.ArrayLike): training labels
            y_test (npt.ArrayLike): test labels
        """
        results = []
        # Run iterative training and evaluation
        if self.mol_transformers is not None:
            for transformer in tqdm(self.mol_transformers, desc="Overall progress"):
                config = SimpleConfig((None, None), transformer=transformer)
                mol_pipe = create_pipeline(config=config,
                                           preprocessing=None,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs,
                                           mol_only=True)
                X_train_trans = mol_pipe.fit_transform(X_train)
                X_test_trans = mol_pipe.transform(X_test)

                for algorithm in tqdm(
                    self.ml_algorithms,
                    desc=f"Models with {transformer[0].replace("Transformer", "")}",
                    position=1,
                    leave=False,
                ):
                    if transformer[0] != "MolecularDescriptorTransformer":
                        self._from_descriptors = False
                    else:
                        self._from_descriptors = True

                    config = SimpleConfig(algorithm)
                    pipe = create_pipeline(config=config,
                                           preprocessing=self.preprocessing,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs)
                    self._data_pipelines.append(pipe)
                    self._steps.append(self._get_steps(mol_pipe, pipe))
                    scores = self._run_evaluation(
                        X_train_trans,
                        X_test_trans,
                        y_train,
                        y_test,
                    )
                    res = {
                        "Algorithm": algorithm[0],
                        "Molecular Transformer": transformer[0]
                    }
                    res.update(scores)
                    results.append(res)
        else:
            for algorithm in tqdm(
                self.ml_algorithms, desc="Models", position=1, leave=False
            ):
                config = SimpleConfig(algorithm)
                pipe = create_pipeline(config=config,
                                       preprocessing=None,
                                       random_state=self.random_state,
                                       n_jobs=self.n_jobs)
                self._data_pipelines.append(pipe)
                self._steps.append(pipe)
                scores = self._run_evaluation(X_train, X_test, y_train, y_test)
                res = {"Algorithm": algorithm[0]}
                res.update(scores)
                results.append(res)

        # store results and select best performing pipeline
        self.results_ = pd.DataFrame(results)
        self._select_best_pipeline()

    def _score_from_predictor(
        self, estimator: Pipeline, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> np.ndarray:
        """Assess performance of given estimator on the provided dataset using selected
        scoring metrics.

        Args:
            estimator (Pipeline): pipeline containing an ML model
            X (npt.ArrayLike): features
            y (npt.ArrayLike): labels

        Returns:
            np.ndarray: performance scores
        """
        y_pred = estimator.predict(X)

        calc_scores = {}
        for scorer in self.scorers:
            value = scorer[1](y, y_pred)
            calc_scores[scorer[0]] = value

        return calc_scores

    def _set_estimators(self):
        """Help to set all estimators from user input (attributes ml_algorithms
        and mol_transformers set using required format).
        """
        methods = [self.ml_algorithms, self.mol_transformers]
        names = ["ml_algorithms", "mol_transformers"]
        full_lists = [REGRESSORS, MOL_TRANSFORMERS]
        for method, name, full_list in zip(methods, names, full_lists):
            if method == "all":
                setattr(self, name, full_list)
            elif method is None and name == "mol_transformers":
                pass
            else:
                custom_list = self._set_custom_estimators(method, full_list)
                setattr(self, name, custom_list)

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
                if name == "r2":
                    results["1-" + name] = 1 - results[name]
                    cols_selection.append("1-" + name)
                else:
                    cols_selection.append(name)
            sorting_df = results[cols_selection].mean(axis=1)

        self.best_index_ = sorting_df.sort_values(ascending=True).index[0]
        steps = self._steps[self.best_index_]
        self.best_estimator_ = Pipeline(steps)

    def _get_non_default_params(self) -> Optional[dict]:
        """Help to get custom attributes on defined instance to use in __str__

        Returns:
            Optional[dict]: attributes as key:value pairs if custom attributes. None is
            returned when all attributes are set as default.
        """
        attr = {}
        if self.ml_algorithms != REGRESSORS:
            attr["ml_algorithms"] = [est[1] for est in self.ml_algorithms]
        if self.mol_transformers != MOL_TRANSFORMERS:
            attr["mol_transformers"] = [est[1] for est in self.mol_transformers]
        if self.preprocessing is not None:
            attr["preprocessing"] = self.preprocessing
        if self.random_state != 21:
            attr["random_state"] = self.random_state
        if self.n_jobs != 1:
            attr["n_jobs"] = self.n_jobs
        if self.scorers != [("r2", SCORING.regression["r2"])]:
            attr["scoring"] = [scorer[0] for scorer in self.scorers]

        return attr if attr else None

    def _set_scoring_functions(self, scoring: Optional[list]) -> list:
        """Help define the scoring functions used during model evaluation.

        Args:
            scoring (Optional[list]): names of scoring function to be used.
                                      If None is provided, r2 will be used.

        Raises:
            ValueError: raise error if 'scoring' is not list or None.

        Returns:
            list: scorers (sklearn or custom scoring functions) used for performance
            evaluation.
        """
        if isinstance(scoring, list):
            scorers = []
            for scorer in scoring:
                try:
                    func = SCORING.regression[scorer]
                    scorers.append((scorer, func))
                except KeyError as err:
                    err.add_note(
                        f"{scorer=} not recognized. Please check available scorers:"
                    )
                    err.add_note("from chempred.config import get_scorer_names")
                    err.add_note("print(get_scorer_names())")
                    raise
        elif scoring is None:
            scorers = [("r2", SCORING.regression["r2"])]
        else:
            raise ValueError("'scoring' accept as inputs lists or None")
        return scorers
