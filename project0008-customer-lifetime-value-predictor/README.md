### Description:

Customer Lifetime Value (CLV) estimates the total revenue a business can expect from a customer over the entire relationship. It’s crucial for optimizing marketing strategies, customer segmentation, and ROI calculations. In this project, we build a regression model to predict CLV based on customer behavior data like purchase frequency, recency, and average order value.

- Builds a regression model to predict customer lifetime value
- Uses behavioral features like recency, frequency, average spend
- Evaluates performance using MSE and R²

## Customer Lifetime Value (CLV) Prediction Using Linear Regression

This script demonstrates how to build a simple **Linear Regression model** to predict **Customer Lifetime Value (CLV)** using customer behavioral data. Below is a detailed explanation of each step, including the reasoning and interpretation of results.

### 📦 Dataset Description

The dataset simulates customer metrics and is structured as follows:

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| `CustomerID`    | Unique identifier for each customer       |
| `Recency`       | Days since the last purchase              |
| `Frequency`     | Total number of purchases                 |
| `AvgOrderValue` | Average order value per purchase          |
| `CLV`           | Customer Lifetime Value (Target Variable) |

### 📊 Goal

To build a model that predicts CLV using **Recency**, **Frequency**, and **AvgOrderValue** as input features.

---

### 🔍 Code Breakdown and Reasoning

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

* **Imports**: Brings in necessary libraries for data handling (pandas, numpy), modeling (sklearn), evaluation, and plotting (matplotlib).

```python
data = {
    'CustomerID': range(1, 11),
    'Recency': [10, 40, 5, 30, 60, 2, 22, 55, 8, 35],
    'Frequency': [5, 3, 10, 2, 1, 15, 4, 1, 7, 2],
    'AvgOrderValue': [100, 120, 80, 200, 150, 60, 90, 250, 110, 130],
    'CLV': [500, 360, 800, 400, 150, 900, 360, 250, 770, 390]
}
df = pd.DataFrame(data)
```

* Creates a **synthetic dataset** with 10 customers and 4 features.

```python
X = df[['Recency', 'Frequency', 'AvgOrderValue']]
y = df['CLV']
```

* Splits the dataset into **independent features (X)** and **target variable (y)**.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* Divides the data into **training (70%)** and **testing (30%)** sets.
* `random_state=42` ensures reproducibility.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

* Initializes and trains a **Linear Regression** model using the training data.

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* Predicts CLV on the test set.
* Calculates **Mean Squared Error (MSE)** and **R² Score**:

  * **MSE**: Average squared difference between actual and predicted values.
  * **R² Score**: Indicates how well the features explain the target. Closer to 1 is better.

```python
print("Predicted CLV values:", np.round(y_pred, 2))
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))
```

* Outputs the model's predictions and evaluation metrics.

```python
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Actual vs Predicted CLV")
plt.xlabel("Actual CLV")
plt.ylabel("Predicted CLV")
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots a **scatter plot** comparing actual and predicted CLV values.
* A red dashed line shows the ideal case (perfect predictions).

---

### 📈 Interpretation of Results

* **Predicted CLV values**: Estimated customer lifetime values based on test data.
* **Mean Squared Error**: A lower MSE means better model accuracy. Example: `MSE = 2616.67` indicates some variance in prediction.
* **R² Score**: Indicates goodness of fit. Example: `R² = 0.72` means that 72% of the variability in CLV can be explained by the model.

### ✅ Summary

This notebook demonstrates a basic approach to predicting Customer Lifetime Value using linear regression. It includes model training, testing, evaluation, and visual interpretation.

> Note: This is a very small dataset. In real-world scenarios, a larger and more diverse dataset, along with feature engineering and model tuning, would be necessary for robust predictions.
