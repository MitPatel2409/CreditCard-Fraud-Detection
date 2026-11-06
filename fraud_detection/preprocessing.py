"""Feature preprocessing and the imbalance-handling strategies.

All transformations are wrapped in a single pipeline together with the model.
This guarantees that scaling statistics and resampling are learned from the
training folds only, so no information from a validation fold or from the test
set reaches the model during fitting.
"""

from enum import Enum

import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraud_detection import config


class ImbalanceStrategy(str, Enum):
    """How a model is prevented from ignoring the rare fraud class.

    UNDERSAMPLE
        Within every training fold, legitimate transactions are randomly
        discarded until both classes have the same number of rows. The model
        sees a balanced problem, at the cost of discarding most of the data.

    CLASS_WEIGHT
        Every training row is kept, and errors on the fraud class are weighted
        more heavily in the loss function, in inverse proportion to the class
        frequencies. No information is discarded, but training uses the full
        dataset and is therefore slower.
    """

    UNDERSAMPLE = "undersample"
    CLASS_WEIGHT = "class_weight"

    @property
    def label(self) -> str:
        """Human-readable name used in tables and plot legends."""
        return {
            ImbalanceStrategy.UNDERSAMPLE: "undersampled",
            ImbalanceStrategy.CLASS_WEIGHT: "class-weighted",
        }[self]


def build_amount_transformer() -> SklearnPipeline:
    """Compress and standardise the transaction amount.

    Amounts are strongly right-skewed (most are small, a few reach tens of
    thousands). ``log1p`` compresses that range so that a handful of very large
    amounts do not dominate distance-based models such as the SVM, and the
    scaler then puts the result on the same scale as the PCA features.
    ``log1p`` rather than ``log`` is used because some amounts are exactly zero.
    """
    return make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )


def build_preprocessor() -> ColumnTransformer:
    """Create the column-wise preprocessing step shared by every model.

    - ``Time`` is dropped (see ``config.DROPPED_COLUMNS`` for the reason).
    - ``Amount`` is log-transformed and standardised.
    - The PCA components ``V1`` to ``V28`` are passed through unchanged.

    The output is kept as a DataFrame so that feature names survive the
    pipeline and can be shown in feature-importance plots.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("drop", "drop", list(config.DROPPED_COLUMNS)),
            ("amount", build_amount_transformer(), [config.AMOUNT_COLUMN]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def build_pipeline(estimator: BaseEstimator, strategy: ImbalanceStrategy) -> Pipeline:
    """Assemble preprocessing, optional undersampling and the model.

    The imbalanced-learn ``Pipeline`` is used instead of the scikit-learn one
    because it applies samplers during ``fit`` only. When the pipeline later
    predicts on a validation fold or the test set, the sampler is skipped, so
    the model is always evaluated on the true, highly imbalanced distribution.

    Args:
        estimator: The classifier, already configured for ``strategy``.
        strategy: Whether undersampling is added to the pipeline.
    """
    steps = [("preprocess", build_preprocessor())]
    if strategy is ImbalanceStrategy.UNDERSAMPLE:
        steps.append(("undersample", RandomUnderSampler(random_state=config.RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
