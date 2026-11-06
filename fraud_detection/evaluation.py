"""Performance metrics for imbalanced binary classification.

Accuracy is deliberately not part of the comparison. With 99.83 % legitimate
transactions, a model that never predicts fraud is 99.83 % accurate and catches
nothing. The metrics below all focus on the fraud class instead.
"""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection import config


def predict_scores(model: ClassifierMixin, features: pd.DataFrame) -> np.ndarray:
    """Return a continuous fraud score for every row, higher meaning more suspicious.

    Ranking metrics (ROC-AUC, average precision) and curves need a score rather
    than a hard 0/1 label. Most models expose a probability through
    ``predict_proba``. The SVM is fitted without probability calibration and
    exposes the signed distance to its decision boundary through
    ``decision_function`` instead, which ranks transactions equally well.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute the fraud-focused metrics for one set of predictions.

    - ``recall``: share of frauds that were caught, TP / (TP + FN).
    - ``precision``: share of fraud alerts that were real frauds, TP / (TP + FP).
    - ``f1`` and ``f2``: harmonic means of the two; F2 weights recall twice as much.
    - ``average_precision``: area under the precision-recall curve (threshold-free).
    - ``roc_auc``: area under the ROC curve (threshold-free). Reported for
      completeness; on data this imbalanced it tends to look optimistic because
      the false-positive rate is divided by a very large number of negatives.
    - ``frauds_caught``, ``frauds_missed``, ``false_alarms``: the confusion
      matrix counts, which are easier to reason about in business terms.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=config.THRESHOLD_BETA, zero_division=0),
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "frauds_caught": int(tp),
        "frauds_missed": int(fn),
        "false_alarms": int(fp),
    }


def evaluate_model(model: ClassifierMixin, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Predict with a fitted model and compute every metric on ``features``."""
    return compute_metrics(target, model.predict(features), predict_scores(model, features))


def build_comparison_table(
    models: dict[str, ClassifierMixin],
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Evaluate several fitted models on the same data, best average precision first."""
    rows = {label: evaluate_model(model, features, target) for label, model in models.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    return table.sort_values("average_precision", ascending=False)



def compare_thresholds(
    default_comparison: pd.DataFrame,
    tuned_comparison: pd.DataFrame,
    columns: tuple[str, ...] = ("recall", "precision", "f2", "frauds_caught", "false_alarms"),
) -> pd.DataFrame:
    """Place the default-threshold and tuned-threshold metrics of the same models side by side.

    Only the rows present in ``tuned_comparison`` are kept. Threshold-free
    metrics (average precision, ROC-AUC) are omitted because moving the
    threshold does not change them.
    """
    selected = list(columns)
    return pd.concat(
        {
            "default threshold": default_comparison.loc[tuned_comparison.index, selected],
            "tuned threshold": tuned_comparison[selected],
        },
        axis=1,
    )
