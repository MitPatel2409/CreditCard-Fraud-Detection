"""Central configuration for the credit card fraud detection experiments.

Every constant that influences an experiment is defined in this module. Keeping
them in one place means that a result can be reproduced, or deliberately
changed, without searching through notebook cells for hard-coded values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The Kaggle archive extracts to ``creditcard.csv/creditcard.csv`` (a folder
# that contains the file). Both that layout and a plain ``creditcard.csv`` file
# at the project root are accepted, in this order of preference.
DATA_CANDIDATES = (
    PROJECT_ROOT / "creditcard.csv" / "creditcard.csv",
    PROJECT_ROOT / "creditcard.csv",
)

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Class"
POSITIVE_LABEL = 1  # 1 = fraudulent transaction, 0 = legitimate transaction.

# ``Time`` is the number of seconds since the first transaction in a two-day
# collection window. It describes where a row sits in this particular file,
# not a property of the transaction that would carry over to new data, so it
# is removed before modelling.
DROPPED_COLUMNS = ("Time",)

# ``Amount`` is the only raw monetary feature. The 28 ``V`` columns are already
# the output of a PCA transformation (centred and of comparable scale), so only
# ``Amount`` needs to be transformed.
AMOUNT_COLUMN = "Amount"

# ---------------------------------------------------------------------------
# Reproducibility and evaluation protocol
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Fraction of rows held out for the final, untouched evaluation. The split is
# stratified so that both partitions keep the original fraud rate (~0.17 %).
TEST_SIZE = 0.30

# Number of stratified folds used for hyperparameter search and threshold
# tuning. Five folds leave roughly 70 fraud cases in every validation fold of
# the training partition, which is enough for a stable estimate.
CV_FOLDS = 5

# Metric that the hyperparameter search maximises. Average precision (the area
# under the precision-recall curve) is used instead of recall alone, because a
# model that labels every transaction as fraud reaches a perfect recall while
# being useless. Average precision rewards ranking frauds above legitimate
# transactions across all thresholds and, unlike ROC-AUC, it is not inflated by
# the very large number of true negatives.
TUNING_SCORING = "average_precision"

# Beta of the F-beta score used to choose the final decision threshold. A beta
# of 2 weighs recall twice as heavily as precision, reflecting that a missed
# fraud is costlier than a legitimate transaction that is reviewed by an analyst.
THRESHOLD_BETA = 2.0

# Number of parameter combinations sampled by every randomised search.
RANDOM_SEARCH_ITERATIONS = 25

# ``-1`` uses every available CPU core for cross-validation. Individual models
# are kept single-threaded so that the parallelism happens at one level only.
N_JOBS = -1
