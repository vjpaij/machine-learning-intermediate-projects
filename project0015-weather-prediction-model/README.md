### Description:

A weather prediction model forecasts conditions such as temperature based on historical weather data. In this project, we build a regression model to predict the next day's temperature using recent temperature trends. We use a Linear Regression model trained on synthetic weather data.

- Creates lag features to use recent temperature values for forecasting
- Uses Linear Regression for time series prediction
- Evaluates model with MSE and R² score
- Visualizes predictions over time

## Weather Prediction Model using Linear Regression

This project demonstrates a simple time series-based weather forecasting model using **linear regression**. It uses synthetic temperature data and predicts the next day's temperature based on the past 3 days' temperatures.

### 📦 Libraries Used

* `numpy`: For numerical operations and generating synthetic noise.
* `pandas`: For time series data handling and feature engineering.
* `sklearn.linear_model.LinearRegression`: Linear regression algorithm.
* `sklearn.model_selection.train_test_split`: Splitting dataset into training and test sets.
* `sklearn.metrics`: To evaluate the prediction accuracy.
* `matplotlib.pyplot`: For plotting the results.

---

### 📊 Step-by-Step Explanation

#### 1. **Generate Synthetic Data**

```python
np.random.seed(42)
days = pd.date_range(start='2023-01-01', periods=100, freq='D')
temperature = 20 + np.sin(np.linspace(0, 3 * np.pi, 100)) * 10 + np.random.normal(0, 2, 100)
```

* Creates a 100-day period starting from Jan 1, 2023.
* Temperature follows a sine wave (to simulate seasonality) with added Gaussian noise.

#### 2. **Create DataFrame and Index by Date**

```python
df = pd.DataFrame({'Date': days, 'Temperature': temperature})
df.set_index('Date', inplace=True)
```

* Makes it easier to work with time series.

#### 3. **Feature Engineering**

```python
for i in range(1, 4):
    df[f'Temp_lag_{i}'] = df['Temperature'].shift(i)
```

* Adds lag features: temperature from 1, 2, and 3 days ago.
* These act as predictors for today's temperature.

#### 4. **Drop NaNs Introduced by Shifting**

```python
df.dropna(inplace=True)
```

* First few rows have NaNs due to shifting. These are removed.

#### 5. **Define Features (X) and Target (y)**

```python
X = df[['Temp_lag_1', 'Temp_lag_2', 'Temp_lag_3']]
y = df['Temperature']
```

* X: Temperatures of past 3 days
* y: Today's temperature

#### 6. **Train-Test Split (No Shuffle)**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
```

* `shuffle=False` preserves time order (important for time series).
* 80% for training, 20% for testing.

#### 7. **Train Linear Regression Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

* Fits a simple linear model using the training data.

#### 8. **Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* `mse`: Measures average squared difference between actual and predicted values. Lower is better.
* `r2`: Coefficient of determination. Ranges from -inf to 1. Closer to 1 means better prediction.

#### 9. **Print Evaluation Metrics**

```python
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```

* Displays the model's accuracy in numbers.

#### 10. **Visualization**

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Temperature')
plt.plot(y_test.index, y_pred, label='Predicted Temperature', linestyle='--')
```

* Helps visually compare real vs predicted temperature.
* Useful for spotting trends, over/under predictions.

---

### 📈 Output Explanation

Example Output:

```
Mean Squared Error: 3.45
R² Score: 0.89
```

* **MSE = 3.45**: The average squared error between actual and predicted temperature is around 3.45°C.
* **R² = 0.89**: The model explains 89% of the variance in temperature data.

This indicates that the linear regression model is able to predict temperature quite accurately based on previous 3 days' temperatures.

---

### ✅ Summary

This simple time series regression model provides a foundational approach to weather forecasting using historical lag features. Though synthetic, the structure is applicable to real-world data with additional complexity (seasonality, external features like humidity, pressure, etc.).
