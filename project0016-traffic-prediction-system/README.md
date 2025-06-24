### Description:

A traffic prediction system forecasts future vehicle flow or congestion levels based on historical traffic data. In this project, we build a simple regression-based model that predicts hourly vehicle counts using previous observations. This can support smart transportation planning and congestion management.

- Forecasts vehicle flow using recent history (lag features)
- Uses Linear Regression for hourly time series prediction
- Evaluates predictions with MSE and R²
- Visualizes predicted vs actual traffic data

## Traffic Volume Prediction Using Linear Regression

### Project Overview

This project demonstrates how to use a simple **Linear Regression** model to predict hourly traffic volume based on past observations. The idea is to use the previous 3 hours' traffic counts to forecast the current hour's traffic count.

---

### Step-by-Step Code Explanation

#### 1. **Importing Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

* `numpy`, `pandas`: For data manipulation and synthetic data generation.
* `sklearn.linear_model.LinearRegression`: Machine learning model.
* `sklearn.model_selection.train_test_split`: To split data into training and testing sets.
* `sklearn.metrics`: For model evaluation.
* `matplotlib.pyplot`: For visualization.

#### 2. **Generate Synthetic Hourly Traffic Data**

```python
np.random.seed(42)
hours = pd.date_range(start='2023-01-01 00:00:00', periods=200, freq='H')
traffic_volume = 200 + 100 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 20, 200)
```

* Simulates hourly traffic for \~8 days.
* `200 + 100 * sin(...)`: Mimics daily traffic patterns with peaks and valleys.
* `np.random.normal(0, 20, 200)`: Adds noise to simulate real-world variation.

#### 3. **Create DataFrame and Lag Features**

```python
df = pd.DataFrame({'Timestamp': hours, 'TrafficVolume': traffic_volume})
df.set_index('Timestamp', inplace=True)

for i in range(1, 4):
    df[f'Lag_{i}'] = df['TrafficVolume'].shift(i)
df.dropna(inplace=True)
```

* The DataFrame is indexed by timestamps.
* Lag features `Lag_1`, `Lag_2`, `Lag_3` store traffic values from 1, 2, and 3 hours ago.
* `dropna()` removes rows with NaN values caused by shifting.

#### 4. **Define Features and Target**

```python
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['TrafficVolume']
```

* Features (`X`): traffic from the previous 3 hours.
* Target (`y`): current hour's traffic volume.

#### 5. **Train/Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
```

* `shuffle=False` to maintain time-series order.
* 80% of data for training, 20% for testing.

#### 6. **Train Linear Regression Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

* Fits a linear model to the training data.

#### 7. **Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```

* `Mean Squared Error (MSE)`: Measures average squared difference between actual and predicted values. Lower is better.
* `R² Score`: Proportion of variance explained by the model. Ranges from 0 (bad) to 1 (perfect). Can be negative if model performs worse than mean prediction.

#### 8. **Visualization**

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Traffic')
plt.plot(y_test.index, y_pred, linestyle='--', label='Predicted Traffic')
plt.title("Traffic Volume Prediction")
plt.xlabel("Time")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots actual vs predicted traffic values.
* Useful to visually inspect how well the model tracks real traffic.

---

### Interpretation of Results

* A **low MSE** suggests that the predictions are close to actual values.
* A **high R² score** (e.g., >0.7) indicates that the model explains a large part of the variability in the data.
* If the lines in the plot closely align, it visually confirms good model performance.

---

### Key Takeaways

* Linear Regression can be effective for time-series forecasting when recent history is predictive of future values.
* Feature engineering (like lag creation) is crucial in time-series problems.
* Always maintain temporal order in splitting data to avoid data leakage.

---

### Potential Improvements

* Use more lag features (e.g., last 24 hours).
* Include external variables like day of week, holidays.
* Try more advanced models like ARIMA, LSTM, or XGBoost.
