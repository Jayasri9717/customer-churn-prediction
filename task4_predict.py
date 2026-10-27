import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------
# LOAD DATASET
# --------------------------------

df = pd.read_csv("churnguard_data.csv")

# --------------------------------
# HANDLE MISSING VALUES
# --------------------------------

# InternetService
df["InternetService"] = df["InternetService"].fillna("DSL")

# TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# tenure
df["tenure"] = df["tenure"].fillna(df["tenure"].mean())

# MonthlyCharges
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())

# Contract
df["Contract"] = df["Contract"].replace(r'^\s*$', np.nan, regex=True)
df["Contract"] = df["Contract"].fillna("Month-to-month")

# --------------------------------
# REMOVE EXTRA SPACES
# --------------------------------

df["gender"] = df["gender"].astype(str).str.strip()
df["PaymentMethod"] = df["PaymentMethod"].astype(str).str.strip()

# --------------------------------
# REMOVE DUPLICATES
# --------------------------------

df = df.drop_duplicates()

# --------------------------------
# DROP customerID
# --------------------------------

df = df.drop("customerID", axis=1)

# --------------------------------
# CHECK NULL VALUES
# --------------------------------

print(df.isnull().sum())

# --------------------------------
# CREATE X AND y
# --------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# --------------------------------
# CONVERT TEXT TO NUMBERS
# --------------------------------

X = pd.get_dummies(X, drop_first=True)

# --------------------------------
# SPLIT DATASET
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# CREATE MODEL
# --------------------------------

model = LogisticRegression(max_iter=5000)

# --------------------------------
# TRAIN MODEL
# --------------------------------

model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# --------------------------------
# PREDICTION
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# ACCURACY
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)