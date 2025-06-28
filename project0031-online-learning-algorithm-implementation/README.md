### Description:

An online learning algorithm updates its model incrementally as new data arrives, making it ideal for streaming or real-time systems. In this project, we implement an online logistic regression classifier using SGDClassifier from scikit-learn, which supports learning in mini-batches.

- Implements online/incremental learning with partial_fit
- Simulates streamed training in batches
- Tracks batch-wise model accuracy and final performance

## Online Learning with `SGDClassifier`

This script demonstrates how to implement **online learning** using scikit-learn's `SGDClassifier` with logistic regression. It simulates a streaming scenario by training the model in small batches using `partial_fit`.

---

### 📌 Code Explanation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

* **numpy, pandas**: Libraries for numerical and tabular data processing.
* **SGDClassifier**: Implements stochastic gradient descent for linear models.
* **make\_classification**: Generates synthetic classification data.
* **train\_test\_split**: Splits dataset into training and test sets.
* **accuracy\_score**: Computes accuracy metric.

```python
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* **make\_classification**: Creates a binary classification dataset with 1,000 samples and 10 features.
* **train\_test\_split**: 70% training, 30% testing split with a fixed random seed.

```python
model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1, warm_start=True)
```

* **loss='log\_loss'**: Logistic regression for classification.
* **learning\_rate='optimal'**: Uses a learning rate schedule based on a theoretical optimum.
* **max\_iter=1**: One pass per `partial_fit` call.
* **warm\_start=True**: Prevents the model from resetting during training; continues learning.

```python
batch_size = 100
n_batches = int(np.ceil(len(X_train) / batch_size))
```

* Divides training data into **mini-batches** of size 100 to simulate streaming/online updates.

---

### 🔁 Online Training Loop

```python
for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))
    acc = model.score(X_batch, y_batch)
    print(f"Batch {i+1}/{n_batches} Accuracy: {acc:.2f}")
```

* Iteratively selects batches from the training data.
* `partial_fit` updates the model using each batch.
* **Accuracy** on the current batch is printed after training it.

This loop simulates how an online model learns from data incrementally — useful for large-scale or streaming data scenarios.

---

### ✅ Final Evaluation

```python
y_pred = model.predict(X_test)
print("\nFinal Evaluation on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

* After all training batches are processed, the model is evaluated on the unseen test set.
* The **accuracy score** reflects the model's overall ability to generalize to new data.

---

### 📊 Result Interpretation

* **Per-batch accuracy**: Shows how well the model learns batch-by-batch. Higher accuracy in later batches may indicate learning progress.
* **Final test accuracy**: Gives an unbiased measure of model performance. For example:

```
Final Evaluation on Test Data:
Accuracy: 0.87
```

Means the model correctly predicted 87% of the test labels — a good performance on synthetic data.

---

### 🔍 Why Use This?

* Simulates **real-time training** for systems where data arrives incrementally (e.g., fraud detection, recommendation systems).
* Memory efficient and scalable to large datasets.

---

### ✅ Summary

This code showcases an **online learning paradigm** using scikit-learn with logistic regression. Instead of training all at once, the model updates incrementally using mini-batches, mimicking real-world data streams.

Ideal for scenarios where:

* Data is too big to fit in memory.
* Data is continuously collected.
* Real-time updates are crucial.
