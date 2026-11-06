"""Model definitions and their hyperparameter search spaces.

Each algorithm is described by a ``ModelSpec``: how to build the estimator for a
given imbalance strategy, which hyperparameters to search, and how to search
them. Adding a new algorithm to the comparison therefore means adding one entry
to ``MODEL_SPECS`` and nothing else.

Parameter names carry the ``model__`` prefix because the estimator is the step
called ``model`` inside the pipeline built by ``preprocessing.build_pipeline``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraud_detection import config
from fraud_detection.preprocessing import ImbalanceStrategy


class SearchMethod(str, Enum):
    """How the hyperparameter space of a model is explored.

    GRID
        Every combination is evaluated. Suitable when the space is small and
        made of a few discrete values, as for logistic regression.

    RANDOM
        A fixed number of combinations is sampled from (often continuous)
        distributions. Randomised search finds good settings far more cheaply
        than a grid when there are many parameters, because only a few of them
        usually matter and a grid wastes most evaluations on the others.
    """

    GRID = "grid"
    RANDOM = "random"


EstimatorFactory = Callable[[ImbalanceStrategy, float], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to tune and train one classification algorithm.

    Attributes:
        name: Display name of the algorithm.
        build_estimator: Function that receives the imbalance strategy and the
            ratio of legitimate to fraudulent training rows, and returns an
            unfitted estimator configured for that strategy.
        param_space: Hyperparameter grid (``GRID``) or distributions (``RANDOM``).
        search_method: How ``param_space`` is explored.
        strategies: The imbalance strategies this algorithm is evaluated with.
    """

    name: str
    build_estimator: EstimatorFactory
    param_space: dict[str, Any]
    search_method: SearchMethod
    strategies: tuple[ImbalanceStrategy, ...]


def _balanced_if_weighted(strategy: ImbalanceStrategy) -> str | None:
    """Return the scikit-learn ``class_weight`` value for a strategy.

    ``"balanced"`` weights every class by ``n_samples / (n_classes * n_class_rows)``.
    After undersampling the classes are already equal, so no weighting is needed.
    """
    return "balanced" if strategy is ImbalanceStrategy.CLASS_WEIGHT else None


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def build_logistic_regression(strategy: ImbalanceStrategy, negative_to_positive: float) -> LogisticRegression:
    """Linear baseline model with L2 (ridge) regularisation.

    The ``newton-cholesky`` solver is designed for data with many more rows
    than features, as here (~160,000 rows, 29 features). It converges in a few
    iterations and fits the class-weighted training folds in seconds.

    L1 (lasso) regularisation was evaluated in an earlier run with the
    ``liblinear`` solver. It never improved cross-validated average precision
    by more than 0.001 over L2, yet some class-weighted fits stalled for many
    minutes, stretching that search to 37 minutes. With only 29 decorrelated
    PCA features there is also little for L1 feature selection to remove, so
    only L2 is used.
    """
    return LogisticRegression(
        solver="newton-cholesky",
        max_iter=1_000,
        class_weight=_balanced_if_weighted(strategy),
        random_state=config.RANDOM_STATE,
    )


LOGISTIC_REGRESSION = ModelSpec(
    name="Logistic Regression",
    build_estimator=build_logistic_regression,
    param_space={
        # Inverse regularisation strength: small values constrain the weights
        # more strongly. The range extends to 1e-4 because undersampled folds
        # contain only ~550 rows and benefit from strong regularisation.
        "model__C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    },
    search_method=SearchMethod.GRID,
    strategies=(ImbalanceStrategy.UNDERSAMPLE, ImbalanceStrategy.CLASS_WEIGHT),
)

# ---------------------------------------------------------------------------
# Support vector machine
# ---------------------------------------------------------------------------


def build_support_vector_machine(strategy: ImbalanceStrategy, negative_to_positive: float) -> SVC:
    """Kernel support vector classifier.

    Probability calibration (Platt scaling) is not requested: it would require
    an extra internal cross-validation on every fit. The signed distance to the
    decision boundary (``decision_function``) ranks transactions just as well
    and is what the ROC, precision-recall and threshold computations use.
    """
    return SVC(
        class_weight=_balanced_if_weighted(strategy),
        cache_size=1_000,
        random_state=config.RANDOM_STATE,
    )


SUPPORT_VECTOR_MACHINE = ModelSpec(
    name="Support Vector Machine",
    build_estimator=build_support_vector_machine,
    param_space={
        # Penalty for margin violations: large values fit the training data harder.
        "model__C": loguniform(1e-2, 1e2),
        # Reach of a single training example for the RBF kernel; ignored by "linear".
        "model__gamma": loguniform(1e-4, 1e0),
        "model__kernel": ["rbf", "linear"],
    },
    search_method=SearchMethod.RANDOM,
    # Kernel SVM training time grows between quadratically and cubically with
    # the number of rows. On ~160,000 rows per training fold a single fit takes far too
    # long for cross-validated tuning, so the SVM is evaluated on undersampled
    # folds only (~550 rows per fit).
    strategies=(ImbalanceStrategy.UNDERSAMPLE,),
)

# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def build_decision_tree(strategy: ImbalanceStrategy, negative_to_positive: float) -> DecisionTreeClassifier:
    """Single decision tree: interpretable, and indifferent to feature scaling."""
    return DecisionTreeClassifier(
        class_weight=_balanced_if_weighted(strategy),
        random_state=config.RANDOM_STATE,
    )


DECISION_TREE = ModelSpec(
    name="Decision Tree",
    build_estimator=build_decision_tree,
    param_space={
        "model__criterion": ["gini", "entropy"],
        # Depth and leaf size are the main guards against over-fitting: an
        # unrestricted tree memorises the training frauds and generalises poorly.
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 20, 50],
    },
    search_method=SearchMethod.GRID,
    strategies=(ImbalanceStrategy.UNDERSAMPLE, ImbalanceStrategy.CLASS_WEIGHT),
)

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def build_xgboost(strategy: ImbalanceStrategy, negative_to_positive: float) -> XGBClassifier:
    """Gradient-boosted trees.

    XGBoost does not accept ``class_weight``. Its equivalent is
    ``scale_pos_weight``, which multiplies the loss of every positive (fraud)
    row. The value recommended by the XGBoost documentation is the ratio of
    negative to positive rows, which is what is passed in for ``CLASS_WEIGHT``.
    """
    scale_pos_weight = negative_to_positive if strategy is ImbalanceStrategy.CLASS_WEIGHT else 1.0
    return XGBClassifier(
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
        random_state=config.RANDOM_STATE,
    )


XGBOOST = ModelSpec(
    name="XGBoost",
    build_estimator=build_xgboost,
    param_space={
        # Number of trees, and how much each tree corrects the previous ones.
        # These two trade off against each other: fewer, larger steps or more,
        # smaller steps.
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(1e-2, 3e-1),
        # Tree complexity.
        "model__max_depth": randint(2, 8),
        "model__min_child_weight": randint(1, 10),
        "model__gamma": uniform(0, 5),
        # Row and column subsampling per tree add randomness that reduces over-fitting.
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        # L2 penalty on leaf weights.
        "model__reg_lambda": loguniform(1e-2, 1e1),
    },
    search_method=SearchMethod.RANDOM,
    strategies=(ImbalanceStrategy.UNDERSAMPLE, ImbalanceStrategy.CLASS_WEIGHT),
)

# The order of this tuple is the order in which models are tuned and reported.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    LOGISTIC_REGRESSION,
    SUPPORT_VECTOR_MACHINE,
    DECISION_TREE,
    XGBOOST,
)


def describe_model_specs(specs: tuple[ModelSpec, ...] = MODEL_SPECS) -> pd.DataFrame:
    """Tabulate every algorithm with its search method, strategies and tuned hyperparameters."""
    rows = [
        {
            "algorithm": spec.name,
            "search_method": spec.search_method.value,
            "imbalance_strategies": ", ".join(strategy.label for strategy in spec.strategies),
            "tuned_hyperparameters": ", ".join(name.removeprefix("model__") for name in spec.param_space),
        }
        for spec in specs
    ]
    return pd.DataFrame(rows).set_index("algorithm")
