"""Charts used by the notebook.

Every function draws on a Matplotlib ``Axes`` (created when none is given) and
returns it, so charts can be combined into larger figures. Colours follow a
fixed assignment: each algorithm always keeps the same colour, whichever chart
it appears in and however many other algorithms are shown with it.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from fraud_detection.evaluation import predict_scores
from fraud_detection.tuning import ThresholdResult

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

# Categorical colours in their fixed order (blue, orange, aqua, yellow, ...).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

ALGORITHM_COLORS = {
    "Logistic Regression": CATEGORICAL[0],
    "Support Vector Machine": CATEGORICAL[1],
    "Decision Tree": CATEGORICAL[2],
    "XGBoost": CATEGORICAL[3],
}

CLASS_COLORS = {"Legitimate (0)": CATEGORICAL[0], "Fraud (1)": CATEGORICAL[1]}

# Single-hue sequential ramp (light to dark blue) for confusion-matrix cells.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("sequential_blue", ["#cde2fb", "#5598e7", "#0d366b"])

CLASS_NAMES = ("Legitimate", "Fraud")


def apply_chart_style() -> None:
    """Set Matplotlib defaults: light surface, recessive grid and axes, thin lines."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS_LINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        }
    )


def _new_axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return ``ax`` unchanged, or a fresh Axes when ``ax`` is ``None``."""
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def algorithm_color(label: str) -> str:
    """Return the fixed colour of the algorithm a label such as ``"XGBoost (...)"`` refers to."""
    for algorithm, color in ALGORITHM_COLORS.items():
        if label.startswith(algorithm):
            return color
    raise KeyError(f"No colour is assigned to the algorithm in {label!r}.")


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------


def plot_class_distribution(class_balance: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Bar chart of the number of transactions per class, labelled with counts and shares.

    The y-axis is linear on purpose: the barely visible fraud bar *is* the finding.

    Args:
        class_balance: Table produced by ``data.describe_class_balance``.
    """
    ax = _new_axes(ax, figsize=(6, 4))
    labels = class_balance.index.tolist()
    counts = class_balance["transactions"].to_numpy()
    ax.bar(labels, counts, width=0.55, color=[CLASS_COLORS[label] for label in labels])
    for position, (count, share) in enumerate(zip(counts, class_balance["share_percent"])):
        ax.annotate(
            f"{count:,}  ({share:.2f} %)",
            xy=(position, count),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=INK_PRIMARY,
        )
    ax.set_title("Transactions per class")
    ax.set_ylabel("Transactions")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="x", visible=False)
    return ax


def plot_amount_distribution(transactions: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Histogram of ``log1p(Amount)`` per class, normalised so both classes are comparable."""
    ax = _new_axes(ax, figsize=(7, 4))
    bins = np.linspace(0, np.log1p(transactions["Amount"].max()), 50)
    for value, label in enumerate(CLASS_COLORS):
        amounts = np.log1p(transactions.loc[transactions["Class"] == value, "Amount"])
        ax.hist(amounts, bins=bins, density=True, histtype="step", linewidth=2, color=CLASS_COLORS[label], label=label)
    ax.set_title("Transaction amount by class")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str, ax: Axes | None = None) -> Axes:
    """Draw a 2x2 confusion matrix with the count written in every cell.

    Rows are the true classes and columns the predicted classes, so the
    bottom-left cell holds the missed frauds and the top-right cell the false
    alarms.
    """
    ax = _new_axes(ax, figsize=(4, 4))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # A logarithmic colour scale keeps the small fraud cells distinguishable
    # from each other next to the ~85,000 true negatives.
    ax.imshow(np.log1p(matrix), cmap=SEQUENTIAL_BLUE)
    midpoint = np.log1p(matrix).max() / 2
    for row, column in itertools.product(range(2), range(2)):
        ax.text(
            column,
            row,
            f"{matrix[row, column]:,}",
            ha="center",
            va="center",
            fontsize=12,
            color="#ffffff" if np.log1p(matrix[row, column]) > midpoint else INK_PRIMARY,
        )
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_confusion_matrices(
    models: dict[str, ClassifierMixin],
    features: pd.DataFrame,
    target: pd.Series,
    columns: int = 4,
) -> plt.Figure:
    """Draw one confusion matrix per model in a grid."""
    rows = int(np.ceil(len(models) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for ax, (label, model) in zip(axes.flat, models.items()):
        plot_confusion_matrix(target, model.predict(features), title=label, ax=ax)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    figure.tight_layout()
    return figure


def plot_threshold_confusion_grid(
    model: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> plt.Figure:
    """Show how the confusion matrix changes as the probability cut-off moves.

    A transaction is labelled as fraud when its predicted probability is at
    least the threshold. Lower thresholds catch more frauds (higher recall) and
    raise more false alarms (lower precision).
    """
    probabilities = predict_scores(model, features)
    columns = 3
    rows = int(np.ceil(len(thresholds) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for ax, threshold in zip(axes.flat, thresholds):
        predictions = (probabilities >= threshold).astype(int)
        caught = predictions[target.to_numpy() == 1].mean()
        plot_confusion_matrix(target, predictions, title=f"Threshold {threshold:.1f}  -  recall {caught:.1%}", ax=ax)
    for ax in axes.flat[len(thresholds):]:
        ax.set_visible(False)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Ranking curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    models: dict[str, ClassifierMixin],
    features: pd.DataFrame,
    target: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """Overlay the ROC curves of several models; the dashed diagonal is random guessing."""
    ax = _new_axes(ax, figsize=(6, 6))
    for label, model in models.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(target, predict_scores(model, features))
        ax.plot(false_positive_rate, true_positive_rate, color=algorithm_color(label), label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=INK_MUTED, label="Random guessing")
    ax.set_title("ROC curve (test set)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    return ax


def plot_precision_recall_curves(
    models: dict[str, ClassifierMixin],
    features: pd.DataFrame,
    target: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """Overlay the precision-recall curves of several models.

    The dashed line is the precision of a model that flags transactions at
    random: it equals the fraud rate, which is why it sits close to zero.
    """
    ax = _new_axes(ax, figsize=(6, 6))
    for label, model in models.items():
        precision, recall, _ = precision_recall_curve(target, predict_scores(model, features))
        ax.plot(recall, precision, color=algorithm_color(label), label=label)
    fraud_rate = target.mean()
    ax.axhline(fraud_rate, linestyle="--", linewidth=1, color=INK_MUTED, label=f"Random guessing ({fraud_rate:.2%})")
    ax.set_title("Precision-recall curve (test set)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left")
    return ax


# ---------------------------------------------------------------------------
# Threshold tuning and model inspection
# ---------------------------------------------------------------------------


def plot_threshold_tuning_curve(result: ThresholdResult, ax: Axes | None = None) -> Axes:
    """Plot the out-of-fold F2 score against the decision threshold.

    The marker shows the selected threshold and the dotted line the model's
    default threshold. Probability thresholds are drawn on a logit axis, which
    stretches the regions close to 0 and 1. Without it, the optimum of a
    class-weighted model (often above 0.99) would be squeezed against the
    right edge of the chart.
    """
    ax = _new_axes(ax, figsize=(5, 3.5))
    color = algorithm_color(result.label)
    thresholds = result.curve["threshold"].to_numpy()
    if result.uses_probabilities:
        thresholds = np.clip(thresholds, 1e-6, 1 - 1e-6)
        ax.set_xscale("logit")
    ax.plot(thresholds, result.curve["f_beta"], color=color)
    best_x = np.clip(result.threshold, 1e-6, 1 - 1e-6) if result.uses_probabilities else result.threshold
    ax.plot(best_x, result.cv_f_beta, marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(
        f"threshold {result.threshold:.4g}\nF2 {result.cv_f_beta:.3f}",
        xy=(best_x, result.cv_f_beta),
        xytext=(0, -38),
        textcoords="offset points",
        ha="center",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.axvline(result.default_threshold, linestyle=":", linewidth=1, color=INK_MUTED)
    ax.set_ylim(0, 1)
    ax.set_title(result.label, fontsize=10)
    ax.set_xlabel("Probability threshold (logit scale)" if result.uses_probabilities else "Decision-function threshold")
    ax.set_ylabel("Out-of-fold F2")
    return ax


def plot_feature_importance(pipeline, label: str, top_n: int = 15, ax: Axes | None = None) -> Axes:
    """Horizontal bar chart of the most important features of a tree-based pipeline.

    For XGBoost and decision trees, importance measures how much each feature
    reduces the training loss across all of its splits. It shows what the model
    relies on, not whether the relationship is causal.
    """
    ax = _new_axes(ax, figsize=(6, 5))
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(pipeline.named_steps["model"].feature_importances_, index=names)
    top = importances.nlargest(top_n).sort_values()
    ax.barh(top.index, top.to_numpy(), height=0.6, color=algorithm_color(label))
    ax.set_title(f"Top {top_n} features - {label}", fontsize=10)
    ax.set_xlabel("Importance (share of total)")
    ax.grid(axis="y", visible=False)
    return ax
