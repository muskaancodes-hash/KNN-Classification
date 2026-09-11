import pandas as pd

# Load Iris dataset
df = pd.read_csv("Iris.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())