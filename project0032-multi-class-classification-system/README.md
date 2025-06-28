### Description:

A multi-class classification system categorizes data into more than two classes. It’s widely used in document classification, image labeling, and customer intent prediction. In this project, we implement a Logistic Regression classifier for multi-class classification on a synthetic dataset using scikit-learn.

- Handles multi-class classification with Logistic Regression
- Uses synthetic data to simulate real-world scenarios
- Evaluates performance using classification metrics and confusion matrix

## Multi-Class Classification using Logistic Regression

This project demonstrates how to perform **multi-class classification** using **Logistic Regression** with a **synthetic dataset** created using `sklearn.datasets.make_classification`. The workflow includes data generation, model training, prediction, and performance evaluation through reports and a confusion matrix.

---

### Code Explanation

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

These are the required libraries:

* `numpy`, `pandas`: general-purpose numerical and data manipulation libraries.
* `make_classification`: used to generate a synthetic dataset.
* `LogisticRegression`: implements logistic regression model.
* `train_test_split`: splits data into training and test sets.
* `classification_report`, `confusion_matrix`: evaluation metrics.
* `matplotlib`, `seaborn`: for plotting.

---

```python
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, 
                           n_classes=4, n_clusters_per_class=1, random_state=42)
```

Generates a **synthetic dataset**:

* `n_samples=500`: 500 data points.
* `n_features=10`: each data point has 10 features.
* `n_informative=6`: 6 features carry useful information.
* `n_classes=4`: 4 target classes (multi-class problem).
* `random_state=42`: ensures reproducibility.

---

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Splits data into:

* 70% training (`X_train`, `y_train`)
* 30% testing (`X_test`, `y_test`)
* Random seed for consistency.

---

```python
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
```

* Initializes a **Logistic Regression** model configured for **multi-class classification** using the **multinomial** loss.
* Uses **'lbfgs'** optimizer (efficient for multinomial problems).
* `max_iter=1000`: ensures enough iterations for convergence.
* Fits model to training data.

---

```python
y_pred = model.predict(X_test)
```

* Predicts target class for each test data point using the trained model.

---

```python
print(classification_report(y_test, y_pred))
```

### Classification Report

Generates a performance summary:

* **Precision**: Correctly predicted positive observations divided by total predicted positives.
* **Recall**: Correctly predicted positives divided by all actual positives.
* **F1-score**: Harmonic mean of precision and recall.
* **Support**: Number of actual instances for each class in the test set.

The report gives these scores **per class**, along with **macro** and **weighted** averages:

* **Macro avg**: Unweighted mean across all classes.
* **Weighted avg**: Mean weighted by class support.

---

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Set2')
plt.title("Confusion Matrix - Multi-Class Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
```

### Confusion Matrix

* Shows actual vs predicted class counts.
* Diagonal cells: correct predictions.
* Off-diagonal: misclassifications.
* `seaborn.heatmap`: creates a colored matrix for easy visual inspection.

---

### Interpretation of Output

* A good model will have **high precision and recall for each class**, and the **confusion matrix** will be mostly diagonal.
* If precision/recall is low for some class, the model may struggle to distinguish that class.
* The confusion matrix helps identify which classes are often confused.

---

### Summary

This script demonstrates a complete multi-class classification pipeline using logistic regression:

1. **Data Preparation**
2. **Model Training**
3. **Prediction**
4. **Evaluation**

It can be extended by trying different classifiers (e.g., Random Forest, SVM), adding feature scaling, or tuning hyperparameters.

---
