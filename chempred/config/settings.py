"""
Module to define available algorithms for ML modeling, class balancing, and molecular
transformers (as implemented in sklearn, imblearn, and scikit-mol, respectively).

@author: Dr. Freddy A. Bernal
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleConfig:
    estimator: tuple
    sampler: Optional[tuple] = None
    transformer: Optional[tuple] = None


@dataclass(frozen=True)
class Defaults:
    classifiers: tuple[str]
    regressors: tuple[str]
    samplers: tuple[str]
