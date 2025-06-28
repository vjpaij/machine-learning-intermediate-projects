### Description:

Self-Organizing Maps (SOMs) are a type of unsupervised neural network that project high-dimensional data onto a low-dimensional (usually 2D) grid. This is useful for clustering, visualization, and understanding complex patterns. In this project, we use MiniSom, a popular Python package for SOMs, to cluster tabular data.

- Implements a Self-Organizing Map for dimensionality reduction and clustering
- Visualizes how data points cluster and organize on a 2D grid
- Works with any high-dimensional tabular dataset

### Self-Organizing Map (SOM) on Iris Dataset - Code Explanation and Analysis

This script demonstrates how to apply a **Self-Organizing Map (SOM)** to the classic **Iris dataset** using the `MiniSom` library. A SOM is an unsupervised learning algorithm used for clustering and dimensionality reduction. Here's a breakdown of the code and what it achieves.

---

## Code Breakdown and Explanation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
```

* **Imports necessary libraries**:

  * `MiniSom` is used for training the SOM.
  * `load_iris` loads the Iris dataset.
  * `MinMaxScaler` normalizes the input features between 0 and 1.

```python
# Load sample dataset (Iris)
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
```

* Loads the Iris dataset into `X` (features) and `y` (target class labels).
* `labels` contains human-readable names of the flower species.

```python
# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

* Feature values are scaled between 0 and 1 for efficient SOM training.

```python
# Initialize Self-Organizing Map
som_size = 10  # 10x10 grid
som = MiniSom(x=som_size, y=som_size, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=100)
```

* A **10x10 SOM grid** is created, where each cell (neuron) is a vector of the same dimension as the input features (4 for Iris).
* `sigma` controls the neighborhood size (how much surrounding neurons are influenced).
* `learning_rate` determines the speed of learning.
* `train_random()` runs the training for 100 iterations with randomly picked samples.

```python
# Plot SOM with labels
plt.figure(figsize=(8, 8))
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Get winning node
    plt.text(w[0], w[1], str(y[i]), color=plt.cm.Set1(y[i] / 3.0), fontdict={'weight': 'bold', 'size': 12})
plt.title("Self-Organizing Map (Iris Data)")
plt.xlim([0, som_size])
plt.ylim([0, som_size])
plt.grid()
plt.tight_layout()
plt.show()
```

* **Each data point is plotted on the SOM grid** at the location of its best-matching unit (BMU) or "winning neuron."
* The **class label (`y[i]`)** is used to color and annotate the corresponding neuron.
* Visualizing these labels shows how well SOM has clustered similar samples together.

---

## Result / Score / Prediction Explanation

* **There is no traditional score (like accuracy)** because SOM is an unsupervised model. However, the plot provides insights:

### What the Plot Shows:

* Each number (`0`, `1`, or `2`) represents one of the three Iris species.
* Clusters form where similar numbers appear close together.
* This indicates the SOM has **learned to organize similar data points** spatially.

### Interpretation:

* If the same class labels (e.g., all `0`s) appear in adjacent neurons, it shows the SOM has found a **meaningful structure** in the data.
* **Overlapping or dispersed numbers** indicate overlapping features between classes or room for feature engineering/improvement.

---

## Why Use SOM?

* Dimensionality reduction and visualization of high-dimensional data.
* Useful in clustering, anomaly detection, or as a preprocessing step.
* Unlike PCA, SOM preserves **topological properties** (similar inputs map to nearby neurons).

---

## Summary

This notebook visualizes unsupervised clustering using a SOM on the Iris dataset. It demonstrates the organization of samples based on feature similarity without knowing their labels during training. The final plot helps interpret how well SOM separated the three flower species based on their petal/sepal measurements.

---

