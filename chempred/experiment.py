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
                    used during evaluation. Defaults to None to assign default names.
        """

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
        """Define the names used for the scoring functions when showing results"""
        if scoring is not None:
            self._named_scoring_functions = scoring
        else:
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
        params = {}
        if "random_state" in estimator().get_params().keys():
            params["random_state"] = self.random_state
        if "n_jobs" in estimator().get_params().keys():
            params["n_jobs"] = self.n_jobs
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

    def _select_best_pipeline(self):
        """Define best model using the obtained performance evaluation. Results are
        stored as the attributes best_index_ and best_estimator_
        """
        av = self.results_[self._named_scoring_functions].mean(axis=1)
        self.best_index_ = av.sort_values(ascending=False).index[0]
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

    def _check_implemented_estimator(self, estimator, implemented_estimators):
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
        best_score = self.results_.loc[self.best_index_, self._named_scoring_functions]
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
        return {
            key: float(val)
            for key, val in zip(self._named_scoring_functions, scores)
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
        self._set_estimators()

    def __str__(self):
        name = type(self).__name__
        attr = self._get_attributes()
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
        columns.extend(self._named_scoring_functions)
        columns.append("Time")
        # Run iterative training and evaluation
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
        ba = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = self._roc_auc(estimator, X, y)
        return np.array([ba, f1, auc])

    def _roc_auc(
            self,
            estimator: Pipeline,
            X: npt.ArrayLike,
            y: npt.ArrayLike
    ) -> float:
        """Calculate ROC AUC for dataset using the given estimator.

        Args:
            estimator (Pipeline): pipeline containing an ML model.
            X (npt.ArrayLike): features.
            y (npt.ArrayLike): labels.

        Returns:
            float: ROC AUC value
        """
        try:
            probs = estimator.predict_proba(X)[:, 1]
        except AttributeError:
            probs = estimator.decision_function(X)
        return roc_auc_score(y, probs)

    def _set_scoring_functions(self, scoring: Optional[list]):
        """Help to define names for scoring functions (as will appear on results).

        Args:
            scoring (list | None): names for scoring functions. If None, the default
                                   names will be used.
        """
        if scoring is not None:
            self._named_scoring_functions = scoring
        else:
            self._named_scoring_functions = ["Balanced Accuracy", "F1 score", "ROC AUC"]

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
            else:
                if method is None and name == "mol_transformers":
                    pass
                else:
                    custom_list = self._set_custom_estimators(method, full_list)
                    setattr(self, name, custom_list)

    def _get_attributes(self) -> Optional[dict]:
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
        scores = ["Balanced Accuracy", "F1 score", "ROC AUC"]
        if self._named_scoring_functions != scores:
            attr["scoring"] = self._named_scoring_functions

        return attr if attr else None
