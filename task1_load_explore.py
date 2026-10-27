import pandas as pd

# Load dataset
df = pd.read_csv("churnguard_data.csv")

# Print shape
print("Shape of dataset:")
print(df.shape)

# Print first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Print column names and data types
print("\nDataset info:")
print(df.info())

# Print missing values count
print("\nMissing values:")
print(df.isnull().sum())

# Print duplicate rows count
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Print value counts of Churn column
print("\nChurn value counts:")
print(df["Churn"].value_counts())

# Print unique values in Contract column
print("\nUnique Contract values:")
print(df["Contract"].unique())