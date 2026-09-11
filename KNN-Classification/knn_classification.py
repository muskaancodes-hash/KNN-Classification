import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Train the KNN model using the best K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn.classes_
)

disp.plot()
plt.title("KNN Confusion Matrix")
plt.show()
import numpy as np

# Select two features for decision boundary
X_2d = df[["PetalLengthCm", "PetalWidthCm"]].values
y_2d = pd.factorize(df["Species"])[0]

# Normalize the two features
scaler_2d = StandardScaler()
X_2d = scaler_2d.fit_transform(X_2d)

# Train KNN model
knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_2d, y_2d)

# Create mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict classes for every point in the grid
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y_2d,
    edgecolor="k"
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("KNN Decision Boundary")
plt.show()