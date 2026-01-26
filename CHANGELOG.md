# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-01-26

### Added
- Decision Trees (both in regression and classification mode).
- TomekLinks included as undersampling technique.
- ADASYN included as oversampling technique.
- Exception management added for failing pipelines (when no scaling used).
- Params attribute stores original user arguments to Explorers.
- `PreprocessingConfig` and `ExplorerConfig`.

### Fixed
- Missing `descriptastorus` dependency for `RDKit2DNovartisScaler`.
- `__str__` method improved to better represent an instant of the Explorers.

### Changed
- Options for data preprocessing. It is possible to filter features only or scale them by two different methods. Default for preprocessing is now `None` (no preprocessing).
- Internal methods `_create_pipeline` and `_instantiate_estimator` were refurbished as independent functions in a new `pipeline` module.
- Internal management of scores from numpy array to dict.
- `_set_estimators` was replaced for separate methods to handle ML algorithms, balancing samplers, and molecular transformers.

