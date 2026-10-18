# Customer Churn Prediction

A Python machine learning project that explores customer churn data, trains three classification models, compares their performance, and saves charts and the best-performing model.

## Project structure

```text
Customer_Churn_Prediction/
├── data/
│   └── customer_churn.csv
├── outputs/
│   ├── best_churn_model.joblib
│   ├── churn_distribution.png
│   ├── decision_tree_confusion.png
│   ├── logistic_regression_confusion.png
│   ├── model_comparison.csv
│   ├── monthly_charges.png
│   ├── random_forest_confusion.png
│   └── tenure_churn.png
├── main.py
└── requirements.txt
```

## Requirements

- Python 3.9 or later
- Packages listed in `requirements.txt`: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and joblib

## Setup and run

From the project directory, create and activate a virtual environment, install dependencies, then run the script:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

Run `main.py` from the project directory so it can find `data/customer_churn.csv`. The script creates `outputs/` if needed and overwrites generated output files on each run.

## What the script does

1. Loads the dataset and reports its shape, columns, and missing-value counts.
2. Drops the customer ID, removes duplicate rows, converts `TotalCharges` to numeric, and maps the `Churn` target (`Yes`/`No`) to 1/0. Rows with invalid churn values are removed.
3. Saves exploratory charts for churn distribution, monthly charges by churn, and tenure by churn.
4. Adds average monthly spend and groups contract types into short-term and long-term categories.
5. Splits the data into stratified training and test sets (80/20, random seed 42). Numeric fields are median-imputed and scaled; categorical fields are most-frequent-imputed and one-hot encoded.
6. Trains Logistic Regression, Decision Tree, and Random Forest classifiers with class weighting. It prints accuracy, precision, recall, F1 score, and classification reports, and saves confusion matrices.
7. Sorts model results by F1 score, writes the comparison CSV, and saves the highest-F1 pipeline as `outputs/best_churn_model.joblib`.

## Generated outputs

- `outputs/churn_distribution.png`, `monthly_charges.png`, `tenure_churn.png`: exploratory charts.
- `outputs/*_confusion.png`: confusion matrices for the three classifiers.
- `outputs/model_comparison.csv`: held-out test metrics, sorted by F1 score.
- `outputs/best_churn_model.joblib`: fitted preprocessing and classifier pipeline selected by F1 score.

The current comparison file shows Logistic Regression as the top model by F1 score (about 0.612). Results can change when the data or code changes. Load the saved pipeline with `joblib.load(...)`; prediction input should be a pandas DataFrame with the training feature columns, excluding `Churn` and `customerID`.

## Dataset

`data/customer_churn.csv` contains customer attributes and the `Churn` outcome. The script expects a `Churn` column and uses `MonthlyCharges`, `tenure`, `TotalCharges`, and `Contract` for analysis and feature engineering. `customerID` is treated as an identifier and dropped when present.
