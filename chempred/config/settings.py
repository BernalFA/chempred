"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from dataclasses import dataclass, asdict
from types import MappingProxyType
from typing import Callable, Optional, Union, Literal


@dataclass
class SimpleConfig:
    """Minimum configuration for a single pipeline. It specifies the algorithm,
    balancing sampler, and the molecular transformer to be used in a single pipeline.

    Attributes:
        estimator (tuple): name and ML estimator.
        sampler (tuple, optional): name and imblearn sampler. If None, no class
                                   balancing will be applied.
        transformer (tuple, optional): name and scikit-mol transformer. If None,
                                       raw features will be used.
    """
    estimator: tuple
    sampler: Optional[tuple] = None
    transformer: Optional[tuple] = None


@dataclass
class ExplorerConfig:
    """Full configuration for an Explorer instance (ClassificationExplorer or
    RegressionExplorer). It specifies which algorithms, transformers, class balancing,
    and scoring along with preprocessing and selection criteria to use during
    exploration.

    Attributes:
        ml_algorithms (list | "all"): ML algorithms to include in exploration.
        balancing_samplers (list | 'all' | None): data samplers for class imbalance
                                                  treatment to include.
        mol_transformers (list | 'all' | None): molecular transformers to include in
                                                exploration.
        preprocessing ("StandardScaler" | "NovartisScaler" | "NoScaler" | None): data
                                preprocessing applied before training the model.
        n_jobs (int): number of cpu units for pipeline processing.
        scoring (list | None): names of the scoring functions to use during evaluation.
        select_best_by (str | list): mode of selection of best performing pipeline.
                The name of a particular metrics used in `scoring` or 'average'.
    """
    ml_algorithms: Union[list, Literal["all"]]
    balancing_samplers: Optional[Union[list, Literal["all"]]]
    mol_transformers: Optional[Union[list, Literal["all"]]]
    preprocessing: Optional[Literal["StandardScaler", "NovartisScaler", "NoScaler"]]
    random_state: int
    n_jobs: int
    scoring: Optional[list]
    select_best_by: str

    def to_dict(self) -> dict:
        "Convert arguments to dictionary"
        return asdict(self)


@dataclass(frozen=True)
class Defaults:
    """Utility to define estimators made available for exploration.

    Attributes:
        classifiers (tuple[str]): names of ML algorithms used for classification.
        regressors (tuple[str]): names of ML algorithms used for regression.
        samplers (tuple[str]): names of imblearn class balancing sampling.
    """
    classifiers: tuple[str]
    regressors: tuple[str]
    samplers: tuple[str]


@dataclass(frozen=True)
class Scoring:
    """Utility to define scoring functions made available for performance evaluation.

    Attributes:
        regression (MappingProxyType[str, Callable]): names and ML estimators for
                                                      regression.
        classification (MappingProxyType[str, Callable]): names and ML estimators for
                                                          classification.
    """
    regression: MappingProxyType[str, Callable]
    classification: MappingProxyType[str, Callable]
