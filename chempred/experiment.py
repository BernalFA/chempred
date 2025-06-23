"""
Module containing the Explorer class, which enables perfoming exploratory experiments.

@author: Dr. Freddy A. Bernal
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.pipeline import Pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
try:
    from IPython import get_ipython
    if "IPKernalApp" in get_ipython().config:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm
from tqdm.contrib.itertools import product

from chempred.config import (
    CLASSIFIERS, MOL_TRANSFORMERS, SAMPLING_METHODS, SimpleConfig, SCORERS
)
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


class ClassificationExplorer(BaseExplorer):
    """Facilitate training and performance assessment of a series of automatically
    created pipelines, combining molecular transformations implemented in rdkit and
    provided in scikit-mol, class imbalance sampling techniques implemented in imblearn,
    and machine learning methods implemented in sklearn. Each pipeline is evaluated for
    predictive performance on a separate dataset (test set) using three different
    metrics commonly used in classification problems (balanced accuracy, F1 score, and
    ROC AUC). Afterward, the best performing pipeline (considering an average of the
    calculated metrics) is made available for prediction of labels on unseen molecular
    structures. Scoring on specified datasets is also possible (further evaluation).

    The current implementation uses default values for all the estimators considered,
    except for 'random_state' and 'n_jobs' that can be configured upon instance
    definition.
    """

    def __init__(
            self,
            ml_algorithms: Union[list, Literal["all"]] = "all",
            balancing_samplers: Optional[Union[list, Literal["all"]]] = "all",
            mol_transformers: Optional[Union[list, Literal["all"]]] = "all",
            preprocessing: bool = True,  # automatically turned off if fingerprints
            random_state: int = 21,
            n_jobs: int = 1,
            scoring: Optional[list] = None,
            select_best_by: str = "average"
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
            preprocessing (bool, optional): if True, a data preprocessing pipeline will
                    be applied before training the model. Defaults to True. Only
                    applicable for molecular descriptors.
            n_jobs (int, optional): number of cpu units for pipeline processing (used
                    on algorithms that allows multiprocessing). Defaults to 1.
            scoring (list | None, optional): names given to the scoring functions
                    used during evaluation. Defaults to None to assign default names.
            select_best_by (str | list): mode of selection of best performing pipeline.
                                         The name of a particular metrics used in
                                         'scoring' can be used. Defaults to 'average',
                                         indicating that all the calculated metrics will
                                         be averaged and the highest average value will
                                         be used to define the best model. If a list is
                                         given, those metrics will be averaged.
        """
        super().__init__(
            ml_algorithms=ml_algorithms,
            mol_transformers=mol_transformers,
            preprocessing=preprocessing,
            random_state=random_state,
            n_jobs=n_jobs,
            scoring=scoring
        )
        self.balancing_samplers = balancing_samplers
        self._select_best_by = self._verify_selection_input(select_best_by)
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
            y_test: npt.ArrayLike
    ):
        """Sequentially create and train pipelines with the training dataset, and assess
        performance using the test dataset. During execution, attributes _data_pipelines
        (containing all the created pipelines for data preprocessing and ML modeling)
        and _steps (containing the steps for the full pipeline including molecular
        transformations) are created. After iteration, the best pipeline is chosen,
        making available the attributes best_estimator_ and best_index_.

        Args:
            X_train (npt.ArrayLike): training smiles / features
            X_test (npt.ArrayLike): test smiles / features
            y_train (npt.ArrayLike): training labels
            y_test (npt.ArrayLike): test labels
        """
        # define columns for resulting dataframe
        columns = ["Algorithm", "Balancing method"]
        columns.extend([scorer[0] for scorer in self.scorers])
        columns.append("Time")
        # Run iterative training and evaluation
        if self.mol_transformers is not None:
            columns.insert(2, "Molecular Transformer")
            results = pd.DataFrame(columns=columns)
            for transformer in tqdm(self.mol_transformers,
                                    desc="Overall progress"):
                mol_pipe = self._create_pipeline(transformer)
                X_train_trans = mol_pipe.fit_transform(X_train)
                X_test_trans = mol_pipe.fit_transform(X_test)

                for algorithm, sampler in product(
                    self.ml_algorithms, self.balancing_samplers,
                    desc=f"Models with {transformer[0].replace("Transformer", "")}",
                    position=1, leave=False
                ):
                    if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                        continue
                    self._last_config = SimpleConfig(algorithm, sampler, transformer)
                    # print(self._last_config)
                    if transformer[0] != "MolecularDescriptorTransformer":
                        self.preprocessing = False
                    else:
                        self.preprocessing = True

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
                if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                    continue
                self._last_config = SimpleConfig(algorithm, sampler)
                pipe = self._create_pipeline()
                self._data_pipelines.append(pipe)
                self._steps.append(pipe)
                scores = self._run_evaluation(X_train, X_test, y_train, y_test)
                res = [algorithm[0], sampler[0]] + scores
                results.loc[len(results)] = res

        # store results and select best performing pipeline
        self.results_ = results
        self._select_best_pipeline()
        # delete attribute '_last_config'
        if hasattr(self, "_last_config"):
            delattr(self, "_last_config")

    def _score_from_predictor(
            self,
            estimator: Pipeline,
            X: npt.ArrayLike,
            y: npt.ArrayLike
    ) -> np.ndarray:
        """Assess performance of given estimator on the provided dataset using balanced
        accuracy, F1 score, and ROC AUC.

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

        calc_scores = []
        for scorer in self.scorers:
            if scorer[0] not in ["roc_auc", "prc_auc"]:
                value = scorer[1](y, y_pred)
            else:
                value = scorer[1](y, probs)
            calc_scores.append(value)

        return np.array(calc_scores)

    def _set_estimators(self):
        """Help to set all estimators from user input (attributes ml_algorithms,
        balancing_samplers, and mol_transformers set using required format).
        """
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
            elif method is None and name == "mol_transformers":
                pass
            elif method is None and name == "balancing_samplers":
                setattr(self, name, [(None, None)])
            else:
                custom_list = self._set_custom_estimators(method, full_list)
                setattr(self, name, custom_list)

    def _select_best_pipeline(self):
        """Define best model from obtained performance metrics. Results are stored as
        attributes best_index_ and best_estimator_
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
        if self.preprocessing is False:
            attr["preprocessing"] = self.preprocessing
        if self.random_state != 21:
            attr["random_state"] = self.random_state
        if self.n_jobs != 1:
            attr["n_jobs"] = self.n_jobs
        if self.scorers != [("balanced_accuracy", SCORERS["balanced_accuracy"])]:
            attr["scoring"] = [scorer[0] for scorer in self.scorers]

        return attr if attr else None

    def _verify_selection_input(self, metrics: Union[list, str]) -> list:
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
                    func = SCORERS[scorer]
                    scorers.append((scorer, func))
                except KeyError as err:
                    err.add_note(
                        f"{scorer=} not recognized. Please check available scorers:"
                    )
                    err.add_note("from chempred.config import get_scorer_names")
                    err.add_note("print(get_scorer_names())")
                    raise
        elif scoring is None:
            scorers = [("balanced_accuracy", SCORERS["balanced_accuracy"])]
        else:
            raise ValueError("'scoring' accept as inputs lists or None")
        return scorers
