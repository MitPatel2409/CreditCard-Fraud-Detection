"""Loading the transaction data and creating the train/test partitions."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection import config


@dataclass(frozen=True)
class DatasetSplit:
    """The stratified train/test partitions used throughout the experiments.

    The test partition is created once, before any resampling or tuning, and is
    used only for the final evaluation. Every modelling decision (resampling,
    hyperparameters, decision threshold) is made with the training partition.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def resolve_data_path(candidates: tuple[Path, ...] = config.DATA_CANDIDATES) -> Path:
    """Return the first candidate path that points to an existing CSV file.

    Raises:
        FileNotFoundError: If none of the candidate paths is a file.
    """
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "The credit card dataset was not found. Download it from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at "
        f"one of these locations:\n{searched}"
    )


def load_transactions(path: Path | None = None) -> pd.DataFrame:
    """Read the transaction dataset into a DataFrame.

    Args:
        path: Location of ``creditcard.csv``. When omitted, the standard
            project locations defined in ``config.DATA_CANDIDATES`` are searched.
    """
    return pd.read_csv(path or resolve_data_path())


def split_features_target(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix ``X`` from the binary target ``y``."""
    features = transactions.drop(columns=config.TARGET_COLUMN)
    target = transactions[config.TARGET_COLUMN]
    return features, target


def describe_class_balance(target: pd.Series) -> pd.DataFrame:
    """Summarise how many transactions belong to each class, and their share."""
    counts = target.value_counts().sort_index()
    summary = pd.DataFrame(
        {
            "transactions": counts,
            "share_percent": (counts / counts.sum() * 100).round(3),
        }
    )
    summary.index = summary.index.map({0: "Legitimate (0)", 1: "Fraud (1)"})
    summary.index.name = "class"
    return summary


def make_train_test_split(features: pd.DataFrame, target: pd.Series) -> DatasetSplit:
    """Split the data into stratified training and test partitions.

    Stratification guarantees that the rare fraud class appears in both
    partitions in the same proportion as in the full dataset. Without it, a
    random split of a 0.17 % class could leave the test set with noticeably more
    or fewer frauds, which would make the reported metrics unreliable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        stratify=target,
        random_state=config.RANDOM_STATE,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
