# Credit Card Fraud Detection

This project compares four classifiers (Logistic Regression, SVM, Decision Tree and XGBoost) for detecting fraudulent credit card transactions. Only 0.17 % of the transactions in the dataset are fraudulent. Every model is tuned by stratified cross-validation and has its decision threshold chosen separately. All models are scored on a held-out test set that is never used for any modelling decision.

## Results (test set: 85,443 transactions, 148 frauds)

Each algorithm is shown with its better imbalance strategy and its cross-validated threshold.

| Model | Avg. precision | Recall | Precision | Frauds caught | False alarms |
|---|---|---|---|---|---|
| **XGBoost (class-weighted)** | **0.835** | 0.784 | **0.913** | 116 | **11** |
| Decision Tree (class-weighted) | 0.753 | 0.770 | 0.740 | 114 | 40 |
| Logistic Regression (class-weighted) | 0.703 | 0.797 | 0.720 | 118 | 46 |
| SVM (undersampled) | 0.692 | 0.777 | 0.677 | 115 | 55 |

Main findings:

- Class weighting beat undersampling for every algorithm that was evaluated both ways.
- Tuning the decision threshold reduced false alarms by a factor of 4.5 to 40 for the three models other than XGBoost.

The notebook explains the full reasoning.

## Project structure

```
FraudDetection.ipynb      Narrative, results and charts
fraud_detection/          Implementation, one module per responsibility
  config.py               All experiment constants (seed, split, folds, metrics)
  data.py                 Loading and the stratified train/test split
  preprocessing.py        Feature transforms and the imbalance strategies
  models.py               The four algorithms and their search spaces
  tuning.py               Hyperparameter search and threshold tuning
  evaluation.py           Fraud-focused metrics and comparison tables
  plotting.py             Charts
```

## Running it

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Place it at `creditcard.csv/creditcard.csv` (the layout the Kaggle archive extracts to) or directly at `creditcard.csv` in this folder. The file is 150 MB, so it is excluded from git.
2. Create an environment and install the dependencies:

   ```bash
   python -m venv .venv
   .venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
   pip install -r requirements.txt
   ```

3. Open `FraudDetection.ipynb` with the `.venv` kernel and run all cells. A full run takes about 9 minutes on a 12-core machine. Lower `RANDOM_SEARCH_ITERATIONS` in `fraud_detection/config.py` for a quicker run.

scikit-learn 1.8 or newer is required.
