"""Credit card fraud detection: data loading, preprocessing, tuning and evaluation.

Modules, in the order the notebook uses them:

- ``config``: every constant that controls an experiment.
- ``data``: loading the CSV and creating the stratified train/test split.
- ``preprocessing``: feature transformations and the two imbalance strategies.
- ``models``: the four algorithms and their hyperparameter search spaces.
- ``tuning``: cross-validated hyperparameter search and threshold tuning.
- ``evaluation``: fraud-focused metrics and model comparison tables.
- ``plotting``: the charts shown in the notebook.
"""
