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
    estimator: tuple
    sampler: Optional[tuple] = None
    transformer: Optional[tuple] = None


@dataclass
class ExplorerConfig:
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
    classifiers: tuple[str]
    regressors: tuple[str]
    samplers: tuple[str]


@dataclass(frozen=True)
class Scoring:
    regression: MappingProxyType[str, Callable]
    classification: MappingProxyType[str, Callable]
