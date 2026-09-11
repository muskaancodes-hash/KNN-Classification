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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Select features and target
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature selection and normalization completed.")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Test different values of K
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"K = {k}, Accuracy = {accuracy:.2f}")

# Find the best K
best_k = k_values[accuracies.index(max(accuracies))]

print("\nBest K value:", best_k)
print("Best Accuracy:", max(accuracies))