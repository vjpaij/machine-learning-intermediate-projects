### Description:

An energy consumption predictor forecasts future power usage based on past consumption data and external factors like time or weather. This helps optimize grid operations, billing, and energy-saving strategies. In this project, we build a regression model to predict energy consumption using lag features.

- Predicts future energy usage using lag-based time series features
- Uses Linear Regression for continuous value forecasting
- Visualizes actual vs predicted energy for insights

### Energy Consumption Forecasting using Linear Regression

This project demonstrates a simple time series forecasting approach using lag features and linear regression. The goal is to predict hourly energy consumption based on the values of previous hours.

---

### 1. **Code Explanation**

#### **Imports**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

These import the necessary libraries:

* `numpy` and `pandas`: For data manipulation.
* `matplotlib.pyplot`: For plotting graphs.
* `sklearn`: For linear regression modeling and evaluation.

#### **Simulated Data Creation**

```python
np.random.seed(42)
hours = pd.date_range(start='2023-01-01', periods=200, freq='H')
energy = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 3, 200)
```

* 200 hourly timestamps are generated starting from Jan 1, 2023.
* Energy consumption follows a sinusoidal pattern with noise, mimicking realistic hourly fluctuation.

#### **Create DataFrame**

```python
df = pd.DataFrame({'Timestamp': hours, 'EnergyConsumption': energy})
df.set_index('Timestamp', inplace=True)
```

* The energy data is stored in a DataFrame with the timestamp as the index.

#### **Feature Engineering (Lag Features)**

```python
for lag in range(1, 4):
    df[f'Lag_{lag}'] = df['EnergyConsumption'].shift(lag)
```

* Lag features represent past energy values.
* `Lag_1`, `Lag_2`, and `Lag_3` represent energy consumption from 1, 2, and 3 hours ago respectively.

```python
df.dropna(inplace=True)
```

* Removes initial rows that contain NaN values due to shifting.

#### **Model Preparation**

```python
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['EnergyConsumption']
```

* `X`: Feature matrix containing lag values.
* `y`: Target variable - current hour energy consumption.

#### **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)
```

* Data is split into 75% training and 25% testing.
* `shuffle=False` maintains the time sequence, which is crucial for time series.

#### **Model Training**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

* A linear regression model is trained on the training set.

#### **Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* `y_pred`: Predicted energy values on test data.
* `mse`: Measures average squared error.
* `r2`: Measures how well predictions match the actual values (1 is perfect, 0 means no explanatory power).

#### **Plotting**

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Energy Consumption')
plt.plot(y_test.index, y_pred, linestyle='--', label='Predicted Energy Consumption')
plt.title("Energy Consumption Forecast")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Displays actual vs predicted energy values for visual comparison.

---

### 2. **Results and Interpretation**

#### **Mean Squared Error (MSE)**

Example output:

```
Mean Squared Error: 8.56
```

* On average, the predictions deviate from actual values by a squared error of \~8.56.
* Lower values indicate better accuracy.

#### **R² Score**

```
R² Score: 0.89
```

* 89% of the variance in energy consumption is explained by the model.
* This is considered a strong fit for a simple linear model.

#### **Graph**

* The plot compares actual and predicted values over time.
* A good model shows predicted values closely tracking the actual line.

---

### 3. **Key Takeaways**

* Lag features effectively capture time dependencies in energy data.
* Linear regression, despite its simplicity, can yield strong predictions with good feature engineering.
* Time-based data must not be shuffled to preserve temporal order.
* R² and MSE provide useful metrics to evaluate model accuracy.

---

