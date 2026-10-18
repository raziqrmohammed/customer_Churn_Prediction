
# ============================================
# CUSTOMER CHURN PREDICTION SYSTEM
# ============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. CREATE OUTPUT FOLDER
os.makedirs("outputs", exist_ok=True)

# 2. LOAD DATASET
file_path = "data/customer_churn.csv"

df = pd.read_csv(file_path)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# 3. DATA PREPROCESSING
# Remove irrelevant customer ID column
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert TotalCharges to numeric.
# Some rows contain blank strings.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Convert target column to 0 and 1
df["Churn"] = (
    df["Churn"].astype(str).str.strip().map(
        {"Yes": 1, "No": 0, "1": 1, "0": 0}
    )
)

# Remove rows with invalid target values
df = df.dropna(subset=["Churn"]).copy()
df["Churn"] = df["Churn"].astype(int)

# 4. EXPLORATORY DATA ANALYSIS (EDA)

print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Churn distribution chart
df["Churn"].value_counts().rename(
    index={0: "Stayed", 1: "Left"}
).plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/churn_distribution.png")
plt.close()

# Monthly charges vs churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("outputs/monthly_charges.png")
plt.close()

# Tenure vs churn
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.savefig("outputs/tenure_churn.png")
plt.close()

# 5. FEATURE ENGINEERING
# Average monthly spend can provide another useful feature.
# Avoid division by zero for customers with zero tenure.
if "TotalCharges" in df.columns and "tenure" in df.columns:
    df["AverageMonthlySpend"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, 1)
    )

# Group contract types
if "Contract" in df.columns:
    df["ContractGroup"] = df["Contract"].replace({
        "Month-to-month": "Short-term",
        "One year": "Long-term",
        "Two year": "Long-term"
    })

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 7. PREPROCESSING PIPELINE
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 8. BUILD MACHINE LEARNING MODELS
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# 9. TRAIN AND EVALUATE MODELS
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"outputs/{safe_name}_confusion.png")
    plt.close()

# 10. COMPARE MODELS
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv", index=False
)

# Choose best model by F1 score
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "outputs/best_churn_model.joblib")

print(f"\nBest model by F1 Score: {best_name}")
print("Model saved to outputs/best_churn_model.joblib")
print("Charts and comparison saved in outputs/")
