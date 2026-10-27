import pandas as pd

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("churnguard_data.csv")

# -----------------------------
# TASK 2 CLEANING STEPS
# -----------------------------

# Drop customerID
df = df.drop("customerID", axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Remove extra spaces
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()

# Standardize casing
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()

# Fix Contract column
df["Contract"] = df["Contract"].replace({
    "Monthly": "Month-to-month",
    "month to month": "Month-to-month",
    "1 year": "One year",
    "2 year": "Two year"
})

# Fix InternetService column
df["InternetService"] = df["InternetService"].replace({
    "dsl": "DSL",
    "FiberOptic": "Fiber optic",
    "None": "No"
})

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove invalid tenure rows
df = df[df["tenure"] > 0]

# Remove invalid MonthlyCharges rows
df = df[
    (df["MonthlyCharges"] >= 10) &
    (df["MonthlyCharges"] <= 200)
]

# Fill missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].mean()
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)

df["tenure"] = df["tenure"].fillna(
    int(df["tenure"].median())
)

# -----------------------------
# ENCODE TARGET COLUMN
# Yes = 1
# No = 0
# -----------------------------

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# -----------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------

df = pd.get_dummies(
    df,
    columns=[
        "gender",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ],
    drop_first=True
)

# -----------------------------
# SPLIT X AND y
# -----------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

# -----------------------------
# TRAIN TEST SPLIT
# 80% train
# 20% test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# ACCURACY
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:")
print(accuracy)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stay", "Churn"]
    )
)