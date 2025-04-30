# ChemPred

[![ccds](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

&nbsp;

ChemPred is a tool for automatic evaluation of different machine learning models with focus on chemistry-related tasks. It takes advantage of the pipelining capabilities offered by [`scikit-learn`](https://scikit-learn.org/stable/), [`imbalanced-learn`](https://imbalanced-learn.org/stable/), and [`scikit-mol`](https://github.com/EBjerrum/scikit-mol). Thus, model performance comparisons on pipelines using several molecular featurization are one of its intended uses. 

> [!IMPORTANT]
> In its current implementation, ChemPred does not offer hyperparameter tuning. Models and pipelines are created using default values.


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
# output
> Algorithm                  | Balancing method    | Molecular Transformer           | balanced_accuracy | Time
> LogisticRegression         | RandomUnderSampler  | MACCSKeysFingerprintTransformer | 0.800             | 0.133
> MLPClassifier              | None                | AvalonFingerprintTransformer    | 0.811             | 2.315
> RidgeClassifier            | SMOTE               | AvalonFingerprintTransformer    | 0.766             | 0.282
> GaussianProcessClassifier  | None                | MorganFingerprintTransformer    | 0.768             | 6.099
> RandomForestClassifier     | RandomUnderSampler  | AvalonFingerprintTransformer    | 0.815             | 0.396
```

## License

The content of this repo is licensed under the [MIT license](./LICENSE) conditions.

