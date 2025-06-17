### Description:

Predicting stock prices is a popular application of machine learning in finance. In this project, we build a simple stock price predictor using historical data and a Linear Regression model. The goal is to predict the next day’s closing price based on recent features like previous prices or trends.

- Downloads and processes real-time financial data
- Builds a simple regression model for next-day price prediction
- Visualizes predicted vs actual prices over time

## Stock Price Prediction using Linear Regression

This Python script demonstrates how to fetch historical stock data and use a simple machine learning model (Linear Regression) to predict stock prices based on previous days' closing prices.

### Prerequisites

Install the required packages using pip if they are not already installed:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

### Description of the Code

```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
```

These are the necessary libraries:

* `yfinance`: To fetch historical stock data.
* `pandas` and `numpy`: For data manipulation.
* `scikit-learn`: For machine learning modeling.
* `matplotlib`: For plotting graphs.

```python
ticker = "AAPL"
df = yf.download(ticker, start="2022-01-01", end="2023-01-01")
```

This block fetches historical stock data for Apple Inc. (AAPL) between Jan 1, 2022 and Jan 1, 2023.

```python
df = df[['Close']]
```

Only the 'Close' column (closing prices) is retained for prediction.

```python
N = 5
for i in range(1, N+1):
    df[f'Close_lag_{i}'] = df['Close'].shift(i)
```

Here, features are engineered: for each row, the closing prices from the past `N` days are used as input features.

```python
df.dropna(inplace=True)
```

Rows with missing values (due to shifting) are dropped.

```python
X = df[[f'Close_lag_{i}' for i in range(1, N+1)]]
y = df['Close']
```

The input features `X` are the lagged closing prices; the target `y` is the current day's closing price.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
```

The dataset is split into training and testing sets (80% train, 20% test). `shuffle=False` ensures chronological order is preserved.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

A Linear Regression model is trained on the training data.

```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")
```

Predictions are made on the test set and evaluated using Mean Squared Error (MSE).

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Price')
plt.plot(y_test.index, y_pred, label='Predicted Price', linestyle='--')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

Finally, the actual and predicted prices are plotted to visually evaluate the performance of the model.

---

This script provides a basic framework for time series forecasting using historical stock data and linear regression. For more accurate predictions, consider using advanced models like LSTM, ARIMA, or ensemble methods.
