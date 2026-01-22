"""
Module containing utility functions for unified use during pipeline creation and
evaluation.

@author: Dr. Freddy A. Bernal
"""

from imblearn.pipeline import Pipeline
from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from chempred.config import SimpleConfig
from chempred.preprocessing import (
    RemoveCorrelated, MissingValuesRemover, RDKit2DNovartisScaler
)


def create_pipeline(
        config: SimpleConfig, preprocessing: str, random_state: int, n_jobs: int,
        mol_only: bool = False
) -> Pipeline:
    """Systematically create a transformation pipeline or a data processing and ML
    training pipeline.

    Args:
        config (SimpleConfig): pipeline configuration including estimator, balancing
                               sampler, and molecular transformer.
        preprocessing (str): whether to use preprocessing, including scaling.
        random_state (int): random seed for estimator instantiation.
        n_jobs (int): number of cores to use for model training.
        mol_only (bool): whether to return only molecular transformation pipeline.

    Raises:
        NotImplementedError: when a wrong preprocessing name is given.

    Returns:
        Pipeline: instantiated imblearn/sklearn pipeline.
    """
    steps = []
    if config.transformer is not None:
        steps.extend([
            ("SmilesToMolTransformer", SmilesToMolTransformer()),
            ("Standardizer", Standardizer()),
            (config.transformer[0],
             config.transformer[1]().set_output(transform="pandas")),
        ])
        if mol_only:
            return Pipeline(steps=steps)
    preprocess = [(
        "MissingValuesRemover",
        MissingValuesRemover().set_output(transform="pandas")
    )]
    if preprocessing is not None:
        preprocess.extend([
            ("VarianceThreshold",
             VarianceThreshold().set_output(transform="pandas")),
            ("RemoveCorrelated",
             RemoveCorrelated().set_output(transform="pandas")),
        ])
        if preprocessing == "StandardScaler":
            preprocess.append((preprocessing, StandardScaler()))
        elif preprocessing == "NovartisScaler":
            preprocess.append((preprocessing, RDKit2DNovartisScaler()))
        elif preprocessing == "NoScaler":
            pass
        else:
            raise NotImplementedError(f"{preprocessing} not implemented")

    steps.extend(preprocess)

    if config.sampler is not None:
        if config.sampler[1] is not None:
            steps.append((
                config.sampler[0],
                instantiate_estimator(config.sampler[1], random_state, n_jobs)
            ))

    steps.append((config.estimator[0],
                  instantiate_estimator(config.estimator[1], random_state, n_jobs)))

    return Pipeline(steps=steps)


def instantiate_estimator(estimator, random_state: int, n_jobs: int):
    """Instantiate given estimator after setting up random_state and n_jobs if possible
    (depending on whether the implementation uses them).

    Args:
        estimator: sklearn, imblearn, or scikit-mol estimator.
        random_state (int): random seed for estimator instantiation.
        n_jobs (int): number of cores to use.

    Returns:
        instantiated estimator
    """
    params = {}
    if "random_state" in estimator().get_params().keys():
        params["random_state"] = random_state
    if "n_jobs" in estimator().get_params().keys():
        params["n_jobs"] = n_jobs
    if estimator.__name__ in ["LGBMClassifier", "LGBMRegressor"]:
        params["verbose"] = -1
        return estimator(**params)
    return estimator()
