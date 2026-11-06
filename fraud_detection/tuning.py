"""Cross-validated hyperparameter search and decision-threshold tuning."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    FixedThresholdClassifier,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from fraud_detection import config
from fraud_detection.models import ModelSpec, SearchMethod
from fraud_detection.preprocessing import ImbalanceStrategy, build_pipeline


@dataclass
class TuningResult:
    """Outcome of tuning one algorithm under one imbalance strategy.

    Attributes:
        model_name: Display name of the algorithm.
        strategy: Imbalance strategy the algorithm was trained with.
        best_pipeline: Pipeline refitted on the whole training partition with
            the best hyperparameters found.
        best_params: The selected hyperparameters (``model__`` prefix removed).
        best_cv_score: Mean validation score of ``best_params`` across folds.
        cv_results: One row per evaluated parameter combination.
        search_seconds: Wall-clock duration of the search, including the refit.
    """

    model_name: str
    strategy: ImbalanceStrategy
    best_pipeline: Pipeline
    best_params: dict
    best_cv_score: float
    cv_results: pd.DataFrame
    search_seconds: float

    @property
    def label(self) -> str:
        """Identifier such as ``"XGBoost (class-weighted)"`` for tables and plots."""
        return f"{self.model_name} ({self.strategy.label})"


def build_cross_validator() -> StratifiedKFold:
    """Return the fold generator shared by every search.

    Stratified folds keep the fraud rate equal in every fold. Shuffling with a
    fixed seed avoids folds that follow the time order of the file, while
    keeping the folds identical across models so that their scores are
    directly comparable.
    """
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def negative_to_positive_ratio(target: pd.Series) -> float:
    """Return the number of legitimate rows per fraudulent row in ``target``."""
    positives = int((target == config.POSITIVE_LABEL).sum())
    return (len(target) - positives) / positives


def _build_search(spec: ModelSpec, pipeline: Pipeline) -> GridSearchCV | RandomizedSearchCV:
    """Create the scikit-learn search object that matches ``spec.search_method``."""
    shared_arguments = {
        "estimator": pipeline,
        "scoring": config.TUNING_SCORING,
        "cv": build_cross_validator(),
        "n_jobs": config.N_JOBS,
        "refit": True,
    }
    if spec.search_method is SearchMethod.GRID:
        return GridSearchCV(param_grid=spec.param_space, **shared_arguments)
    return RandomizedSearchCV(
        param_distributions=spec.param_space,
        n_iter=config.RANDOM_SEARCH_ITERATIONS,
        random_state=config.RANDOM_STATE,
        **shared_arguments,
    )


def _readable(value):
    """Round floats sampled from continuous distributions so that tables stay legible."""
    return round(float(value), 4) if isinstance(value, float) else value


def _summarise_cv_results(search: GridSearchCV | RandomizedSearchCV) -> pd.DataFrame:
    """Keep the parameter values and the mean/std validation score, best first."""
    results = pd.DataFrame(search.cv_results_)
    parameter_columns = [column for column in results.columns if column.startswith("param_")]
    summary = results[[*parameter_columns, "mean_test_score", "std_test_score", "rank_test_score"]]
    summary = summary.rename(columns=lambda column: column.removeprefix("param_model__"))
    return summary.sort_values("rank_test_score").reset_index(drop=True)


def tune_model(
    spec: ModelSpec,
    strategy: ImbalanceStrategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> TuningResult:
    """Search the hyperparameters of one algorithm under one imbalance strategy.

    Every candidate is scored with stratified cross-validation on the training
    partition only. Because undersampling is a pipeline step, it is re-applied
    inside each training fold while every validation fold keeps its natural
    fraud rate. The validation scores therefore estimate performance on
    realistic data rather than on an artificially balanced sample.
    """
    estimator = spec.build_estimator(strategy, negative_to_positive_ratio(y_train))
    search = _build_search(spec, build_pipeline(estimator, strategy))

    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    best_params = {
        name.removeprefix("model__"): _readable(value) for name, value in search.best_params_.items()
    }
    return TuningResult(
        model_name=spec.name,
        strategy=strategy,
        best_pipeline=search.best_estimator_,
        best_params=best_params,
        best_cv_score=search.best_score_,
        cv_results=_summarise_cv_results(search),
        search_seconds=elapsed,
    )


def tune_all_models(
    specs: tuple[ModelSpec, ...],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[TuningResult]:
    """Tune every algorithm under every strategy it supports, printing progress."""
    results = []
    for spec in specs:
        for strategy in spec.strategies:
            result = tune_model(spec, strategy, X_train, y_train)
            print(
                f"{result.label:<45} CV {config.TUNING_SCORING} = {result.best_cv_score:.4f}"
                f"   ({result.search_seconds:,.0f} s)"
            )
            results.append(result)
    return results


def summarise_tuning(results: list[TuningResult]) -> pd.DataFrame:
    """Tabulate the best cross-validated score and parameters of every search."""
    rows = [
        {
            "model": result.label,
            f"cv_{config.TUNING_SCORING}": result.best_cv_score,
            "search_seconds": result.search_seconds,
            "best_params": result.best_params,
        }
        for result in results
    ]
    return pd.DataFrame(rows).set_index("model").sort_values(f"cv_{config.TUNING_SCORING}", ascending=False)


def select_best_variant_per_algorithm(results: list[TuningResult]) -> list[TuningResult]:
    """Keep, for every algorithm, the strategy with the best cross-validated score.

    The choice uses the cross-validation score rather than the test score, so
    that the test set is not used to make any modelling decision.
    """
    best: dict[str, TuningResult] = {}
    for result in results:
        incumbent = best.get(result.model_name)
        if incumbent is None or result.best_cv_score > incumbent.best_cv_score:
            best[result.model_name] = result
    return list(best.values())


@dataclass
class ThresholdResult:
    """Outcome of choosing the decision threshold of one tuned pipeline.

    Attributes:
        label: Identifier of the tuned pipeline, e.g. ``"XGBoost (class-weighted)"``.
        classifier: The tuned pipeline (not refitted) wrapped so that ``predict``
            applies ``threshold`` instead of the default cut-off.
        threshold: The selected score threshold. A transaction is labelled as
            fraud when its score is greater than or equal to this value.
        default_threshold: The cut-off ``predict`` would otherwise use
            (0.5 for probabilities, 0 for SVM decision values).
        cv_f_beta: F-beta of ``threshold`` on the out-of-fold training scores.
        curve: Precision, recall and F-beta at every candidate threshold.
    """

    label: str
    classifier: FixedThresholdClassifier
    threshold: float
    default_threshold: float
    cv_f_beta: float
    curve: pd.DataFrame

    @property
    def uses_probabilities(self) -> bool:
        """Whether the threshold applies to a probability (rather than an SVM distance)."""
        return self.classifier.response_method == "predict_proba"


def _response_method(pipeline: Pipeline) -> str:
    """Return the scoring method of a pipeline: probabilities when available."""
    return "predict_proba" if hasattr(pipeline, "predict_proba") else "decision_function"


def _f_beta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """Element-wise F-beta score, defined as 0 where precision and recall are both 0."""
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    numerator = (1 + beta_squared) * precision * recall
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def tune_decision_threshold(
    result: TuningResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ThresholdResult:
    """Choose the decision threshold of a tuned pipeline by cross-validation.

    ``predict`` labels a transaction as fraud when its score passes a default
    cut-off (0.5 for probabilities, 0 for SVM distances). Both imbalance
    strategies make that default too low. A model trained on undersampled
    (50/50) data, or with fraud errors weighted ~580 times, behaves as if fraud
    were far more common than the real 0.17 % and over-estimates every
    transaction's fraud probability. The default threshold therefore raises
    many false alarms.

    The procedure:

    1. Out-of-fold scores are produced for every training row with
       ``cross_val_predict``: each row is scored by a pipeline fitted on the
       other folds, so no row is scored by a model that saw it.
    2. Precision, recall and F-beta are computed at every distinct score, which
       is an exact search rather than a coarse grid of candidate thresholds.
    3. The threshold with the highest F-beta is kept.

    The pipeline that was already fitted on the full training partition is
    reused unchanged (``FrozenEstimator``); only its decision rule changes.
    The test set plays no part in the choice.
    """
    response_method = _response_method(result.best_pipeline)
    out_of_fold_scores = cross_val_predict(
        clone(result.best_pipeline),
        X_train,
        y_train,
        cv=build_cross_validator(),
        method=response_method,
        n_jobs=config.N_JOBS,
    )
    if response_method == "predict_proba":
        out_of_fold_scores = out_of_fold_scores[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, out_of_fold_scores)
    # The last precision/recall pair is a sentinel without a threshold.
    curve = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    curve["f_beta"] = _f_beta(curve["precision"].to_numpy(), curve["recall"].to_numpy(), config.THRESHOLD_BETA)
    best = curve.loc[curve["f_beta"].idxmax()]

    classifier = FixedThresholdClassifier(
        FrozenEstimator(result.best_pipeline),
        threshold=float(best["threshold"]),
        response_method=response_method,
    ).fit(X_train, y_train)
    return ThresholdResult(
        label=result.label,
        classifier=classifier,
        threshold=float(best["threshold"]),
        default_threshold=0.5 if response_method == "predict_proba" else 0.0,
        cv_f_beta=float(best["f_beta"]),
        curve=curve,
    )
