# ChemPred

### Overview

`ChemPred` ([GitHub repository](https://github.com/BernalFA/chempred)) is a Python package comprising a set of tools for automatic evaluation of different machine learning models with focus on chemistry-related tasks. Taking advantage of the pipelining capabilities offered by [`scikit-learn`](https://scikit-learn.org/stable/), [`imbalanced-learn`](https://imbalanced-learn.org/stable/), and [`scikit-mol`](https://github.com/EBjerrum/scikit-mol), `ChemPred` offers performance comparisons on pipelines created using several molecular featurization methods, different machine learning algorithms, and class balancing options (in case of classification problems).

### Installation

`ChemPred` relies on several external Python packages as described above, which are already included as dependencies. Thus, installation from source is possible as usual.

```bash
$ git clone https://github.com/BernalFA/chempred.git
$ cd chempred
$ pip install .
```

### Documentation

The documentation follows the best practice for project documentation as described by *Daniele Procida* in the [Diátaxis documentation framework](https://diataxis.fr/) and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)