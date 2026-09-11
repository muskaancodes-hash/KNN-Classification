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
## Part 3: KNN Classification

A K-Nearest Neighbors (KNN) classifier was implemented using Scikit-learn.

### Steps Performed

* Tested different K values: 1, 3, 5, 7, 9, and 11.
* Trained a KNN model for each K value.
* Predicted the test data.
* Calculated accuracy for each K.
* Identified the K value with the highest accuracy.

### Tool Used

* Scikit-learn
* ## Part 4: Model Evaluation

The KNN model was evaluated using a confusion matrix and classification report.

### Evaluation Methods

* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Accuracy

The confusion matrix was also visualized using Matplotlib to understand the model's correct and incorrect predictions for each Iris species.

