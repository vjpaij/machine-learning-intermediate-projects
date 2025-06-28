### Description:

CatBoost (Categorical Boosting) is a powerful gradient boosting library developed by Yandex, known for handling categorical features natively, delivering high performance with minimal preprocessing. In this project, we implement a CatBoost classifier on tabular data with categorical features and evaluate its performance.

- Implements CatBoost with mixed data types (categorical + numerical)
- Requires minimal preprocessing for categorical features
- Provides accurate, efficient gradient boosting classification

## CatBoost Classifier Example: Code Explanation and Results

This notebook demonstrates how to use CatBoost, a gradient boosting library, for a classification task involving both categorical and numerical features. Below is a detailed explanation of each part of the code, along with the meaning of the output results.

### Libraries Used

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

These are standard libraries:

* `pandas`, `numpy` for data manipulation.
* `catboost` for the classification model.
* `sklearn` for splitting data and evaluating performance.
* `seaborn`, `matplotlib` for visualization.

### Simulated Dataset

```python
data = {
    'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'Bachelors'],
    'Experience': [2, 5, 10, 3, 7, 12, 4, 6, 11, 1],
    'Department': ['HR', 'IT', 'Finance', 'Finance', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance'],
    'Target': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
```

A small dataset is created with:

* **Categorical features**: `Education`, `Department`
* **Numerical feature**: `Experience`
* **Target variable**: `Target` (binary classification: 0 or 1)

### Feature Definition

```python
X = df.drop('Target', axis=1)
y = df['Target']
cat_features = [0, 2]  # 'Education' and 'Department'
```

Here:

* `X` holds features, `y` is the target.
* `cat_features` indicates the indices of categorical columns (required by CatBoost).

### Data Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* Splits the dataset into 70% training and 30% testing data.

### Model Training

```python
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0)
model.fit(X_train, y_train, cat_features=cat_features)
```

* Initializes the CatBoost classifier.
* Trains on `X_train`, `y_train` with specified categorical features.

### Prediction and Evaluation

```python
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
```

* Predicts outcomes for test data.
* Prints a classification report including precision, recall, F1-score, and support for each class:

#### Example Output (may vary based on split):

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         1

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3
```

**Interpretation**:

* **Precision**: Out of predicted class `X`, how many were correct?
* **Recall**: Out of actual class `X`, how many were predicted correctly?
* **F1-Score**: Harmonic mean of precision and recall.
* **Accuracy**: Overall correct predictions.

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - CatBoost Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

* Visualizes how many predictions were correct/incorrect.
* Diagonal elements show correct predictions, off-diagonal are misclassifications.

In our example, the small dataset and model setup often lead to **perfect classification**, but in real-world data, performance depends on various factors like data size, quality, and model tuning.

---

### Summary

This code provides a basic demonstration of:

* Using CatBoost for classification with categorical and numerical features.
* Splitting data and training a model.
* Evaluating results using classification metrics and confusion matrix.

This makes it a great starting point for working with real-world datasets involving mixed data types.
