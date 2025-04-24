# ChemPred

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ChemPred is a tool for automatic evaluation of different machine learning models with focus on chemistry-related tasks. It takes advantage of the pipelining capabilities offered by [`scikit-learn`](https://scikit-learn.org/stable/), [`imbalanced-learn`](https://imbalanced-learn.org/stable/), and [`scikit-mol`](https://github.com/EBjerrum/scikit-mol). Thus, model performance comparisons on several molecular featurization is also included. 

[!IMPORTANT]
In its current implementation, ChemPred does not offer hyperparameter tuning. Models and pipelines are created using default values.


## Usage

Minimum example:

```python
from chempred.experiment import ClassificationExplorer

# set up exploration experiment
experiment = ClassificationExplorer()
# run pipeline creation and evaluation
experiment.evaluate(smiles_train, smiles_test, y_train, y_test)
# check results
experiment.results_.head()
```
```
> Algorithm       | Balancing method   | Molecular Transformer          | Balanced Accuracy | F1 score | ROC AUC | Time
> DummyClassifier | None               | AtomPairFingerprintTransformer | 0.500             | 0.860    | 0.500   | 0.0062
> GausianNB       | RandomUnderSampler | AtomPairFingerprintTransformer | 0.673             | 0.851    | 0.673   | 0.0712
> GausianNB       | SMOTE              | AtomPairFingerprintTransformer | 0.612             | 0.880    | 0.612   | 0.1078
```

## License

The content of this repo is licensed under the [MIT license](./LICENSE) conditions.
--------

