# KNN-Classification
Implementation of K-Nearest Neighbors (KNN) classification using the Iris dataset with feature normalization, different K values, accuracy evaluation, confusion matrix, and decision boundary visualization.
## Part 1: Load Dataset

The Iris dataset was loaded using Pandas.

### Steps Performed

* Loaded `Iris.csv` using Pandas.
* Displayed the first five rows.
* Checked the dataset shape.
* Displayed column names.
* Checked for missing values.

### Tools Used

* Python
* Pandas
* Iris Dataset
## Part 2: Feature Selection and Normalization

The four numerical features of the Iris dataset were selected for KNN classification.

### Steps Performed

* Selected Sepal Length, Sepal Width, Petal Length, and Petal Width.
* Separated features from the target variable.
* Split the dataset into training and testing sets.
* Used StandardScaler to normalize the features.

### Result

* Training samples: 120
* Testing samples: 30
* Number of features: 4
