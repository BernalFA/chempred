from itertools import product
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from chempred.config import SimpleConfig
from chempred.pipeline import create_pipeline, score_from_predictor


def _evaluate_single_pipeline(args):
    config = SimpleConfig(estimator=args["algorithm"], sampler=args["sampler"],
                          transformer=args["transformer"])
    pipe = create_pipeline(
        config=config,
        preprocessing=args["preprocessing"],
        random_state=args["random_state"],
        n_jobs=args["n_jobs"]
    )
    # print(pipe.named_steps)
    try:
        pipe.fit(args["X_train"], args["y_train"])
        scores = score_from_predictor(pipe, X=args["X_test"], y=args["y_test"],
                                      scorers=args["scorers"])
    except ValueError:
        scores = {scorer[0]: np.nan for scorer in args["scorers"]}

    # print(scores)
    res = {
        "algorithm": args["algorithm"][0],
        "sampler": args["sampler"][0],
        "transformer": args["transformer"][0],
    }
    res.update(scores)
    # print(res)
    return res


def evaluate_pipelines_in_parallel(ml_algorithms, balancing_samplers, mol_transformers,
                                   preprocessing, scorers, random_state, n_jobs,
                                   X_train, X_test, y_train, y_test):
    args_list = []
    if mol_transformers is not None:
        combinations = list(
            product(ml_algorithms, balancing_samplers, mol_transformers)
        )
        for algorithm, sampler, transformer in combinations:
            if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                continue
            else:
                args = {
                    "algorithm": algorithm,
                    "sampler": sampler,
                    "transformer": transformer,
                    "preprocessing": preprocessing,
                    "scorers": scorers,
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test
                }
                args_list.append(args)

    else:
        combinations = list(product(ml_algorithms, balancing_samplers))
        for algorithm, sampler, transformer in combinations:
            if algorithm[0] == "DummyClassifier" and sampler[0] is not None:
                continue
            else:
                args = {
                    "algorithm": algorithm,
                    "sampler": sampler,
                    "transformer": (None, None),
                    "preprocessing": preprocessing,
                    "scorers": scorers,
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test
                }
                args_list.append(args)

    results = Parallel(n_jobs=8)(
        delayed(_evaluate_single_pipeline)(args) for args in args_list
    )

    return pd.DataFrame(results)
