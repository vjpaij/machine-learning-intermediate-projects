### Description:

Sales forecasting predicts future sales based on historical trends, helping businesses make informed decisions on inventory, staffing, and promotions. In this project, we simulate monthly sales data and use Linear Regression with lag features to forecast future sales.

- Forecasts future sales using lag-based features
- Applies Linear Regression to time series data
- Visualizes and evaluates results using MSE and R² Score

## Time Series Sales Forecasting using Linear Regression

### Overview

This script demonstrates a basic **time series regression** model using **lag features** to forecast monthly sales data. It leverages `LinearRegression` from scikit-learn and evaluates performance using **Mean Squared Error (MSE)** and **R-squared (R²)** score.

---

### Step-by-Step Explanation

#### 1. **Imports**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

These libraries are used for:

* Data manipulation: `pandas`, `numpy`
* Plotting: `matplotlib`
* Modeling and evaluation: `scikit-learn`

---

#### 2. **Simulate Monthly Sales Data**

```python
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=24, freq='M')
sales = 1000 + np.sin(np.linspace(0, 3 * np.pi, 24)) * 200 + np.random.normal(0, 50, 24)
```

* Generates 24 months (2 years) of sales data.
* Adds a sinusoidal pattern to simulate seasonality.
* Adds Gaussian noise to simulate randomness.

---

#### 3. **Create DataFrame**

```python
df = pd.DataFrame({'Month': months, 'Sales': sales})
df.set_index('Month', inplace=True)
```

Creates a time series DataFrame with 'Month' as the index and corresponding 'Sales'.

---

#### 4. **Create Lag Features**

```python
for lag in range(1, 4):
    df[f'Lag_{lag}'] = df['Sales'].shift(lag)
```

* Adds lag features for the past 3 months (Lag\_1 = last month, Lag\_2 = 2 months ago, etc.).
* Helps the model learn temporal patterns.

---

#### 5. **Drop Missing Values**

```python
df.dropna(inplace=True)
```

* Drops the first few rows that contain `NaN` due to lagging.

---

#### 6. **Feature and Target Separation**

```python
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['Sales']
```

* `X`: past 3 months' sales
* `y`: current month's sales

---

#### 7. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)
```

* Time series split (no shuffling!)
* 75% data for training, 25% for testing

---

#### 8. **Train the Linear Regression Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

* Trains the model on historical sales lag features.

---

#### 9. **Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* Predicts sales on test data.
* `MSE`: average squared difference between predicted and actual sales.
* `R² Score`: how well the model explains the variance in the data (closer to 1 is better).

Example Output:

```
Mean Squared Error: 3165.77
R² Score: 0.83
```

* This means predictions are reasonably close to actual sales, and the model explains 83% of the variance.

---

#### 10. **Visualization**

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Sales')
plt.plot(y_test.index, y_pred, label='Predicted Sales', linestyle='--')
plt.title("Sales Forecasting using Time Series Regression")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots actual vs predicted sales to visually assess model performance.
* Dotted line represents the model's forecast.

---

### Conclusion

This notebook demonstrates how simple linear regression with lag features can be used for time series forecasting. While effective for linear relationships, more advanced methods (e.g., ARIMA, LSTM, XGBoost) may yield better performance for complex series.
