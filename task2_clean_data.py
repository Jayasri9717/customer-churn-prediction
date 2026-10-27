import pandas as pd

# Load dataset
df = pd.read_csv("churnguard_data.csv")

# Drop customerID column
df = df.drop("customerID", axis=1)

# Remove duplicate rows
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
    "Month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "Monthly": "Month-to-month",
    "One year": "One year",
    "1 year": "One year",
    "Two year": "Two year",
    "2 year": "Two year"
})

# Fix InternetService column
df["InternetService"] = df["InternetService"].replace({
    "dsl": "DSL",
    "DSL": "DSL",
    "Fiber optic": "Fiber optic",
    "FiberOptic": "Fiber optic",
    "None": "No",
    "no": "No"
})

# Convert TotalCharges to numeric
df["InternetService"] = df["InternetService"].fillna("No")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove rows where tenure <= 0
df = df[df["tenure"] > 0]

# Remove rows where MonthlyCharges < 10 or > 200
df = df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200)]

# Fill missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

df["tenure"] = df["tenure"].fillna(int(df["tenure"].median()))

# Print cleaned dataframe shape
print("Cleaned DataFrame shape:")
print(df.shape)
#Fill missing values in InternetService 
df["InternetService"] = df["InternetService"].fillna("DSL")

# Print missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())