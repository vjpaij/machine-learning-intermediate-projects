### Description:

A price optimization tool helps businesses determine the ideal price point for products to maximize revenue or profit. In this project, we simulate a dataset of product prices and sales, and build a regression model to understand the price-demand relationship and identify the optimal price for maximum revenue.

- Simulates price vs demand and calculates revenue
- Builds a linear model to estimate the price-demand curve
- Identifies the optimal price that maximizes predicted revenue
- Visualizes the revenue curve and optimal price point

### Price Optimization using Linear Regression

This script simulates a real-world price optimization scenario. It demonstrates how to use linear regression to predict product demand based on price and identify the price point that maximizes revenue.

---

## 1. **Imports and Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

* `numpy` and `pandas` are used for data manipulation.
* `matplotlib.pyplot` is used for plotting.
* `LinearRegression` from `scikit-learn` is used to train a regression model.

---

## 2. **Simulated Data Generation**

```python
np.random.seed(42)
prices = np.linspace(5, 50, 20)
demand = 1000 - 15 * prices + np.random.normal(0, 50, len(prices))
revenue = prices * demand
```

* Prices range from \$5 to \$50.
* Demand is modeled as decreasing linearly with price (`1000 - 15 * price`), with added Gaussian noise.
* Revenue is simply `price × demand`.

---

## 3. **DataFrame Creation**

```python
df = pd.DataFrame({'Price': prices, 'Demand': demand, 'Revenue': revenue})
```

Creates a DataFrame with three columns: `Price`, `Demand`, and `Revenue`.

---

## 4. **Training Linear Regression Model**

```python
X = df[['Price']]
y = df['Demand']
model = LinearRegression()
model.fit(X, y)
```

* `X`: independent variable (Price)
* `y`: target variable (Demand)
* Fits a linear regression model to learn the relationship between price and demand.

---

## 5. **Predicting Demand and Revenue**

```python
price_range = np.linspace(5, 50, 100).reshape(-1, 1)
predicted_demand = model.predict(price_range)
predicted_revenue = price_range.flatten() * predicted_demand
```

* Generates a finer range of prices (\$5 to \$50).
* Uses the trained model to predict demand for each price.
* Computes the corresponding predicted revenue.

---

## 6. **Finding the Optimal Price**

```python
optimal_index = np.argmax(predicted_revenue)
optimal_price = price_range.flatten()[optimal_index]
max_revenue = predicted_revenue[optimal_index]
```

* Finds the price with the highest predicted revenue.
* This is the **optimal price** for maximizing revenue.

---

## 7. **Printing and Plotting Results**

```python
print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Expected Revenue at Optimal Price: ${max_revenue:.2f}")
```

* Outputs:

  * `Optimal Price`: The price point where revenue is highest.
  * `Expected Revenue`: The predicted revenue at that price.

```python
plt.figure(figsize=(8, 5))
plt.plot(price_range, predicted_revenue, label='Predicted Revenue')
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
plt.title("Price Optimization: Revenue vs Price")
plt.xlabel("Price ($)")
plt.ylabel("Predicted Revenue ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots the relationship between price and predicted revenue.
* Highlights the optimal price using a vertical red dashed line.

---

## 🔍 Interpretation of Results

* **Model**: A linear model predicts demand as a function of price.
* **Prediction**: For each possible price, the model estimates demand.
* **Revenue Estimation**: Revenue is computed for each price point.
* **Optimization Goal**: Find the price that yields the maximum predicted revenue.
* **Output Meaning**:

  * `Optimal Price`: Best price to set for maximizing revenue.
  * `Expected Revenue`: How much revenue we can expect at that price.

---

## ✅ Summary

This is a basic price optimization simulation using linear regression. While simplistic, it provides a strong foundation for more advanced techniques like:

* Polynomial regression (if demand vs price is non-linear)
* Price elasticity models
* Real-world demand curves from market data

Ideal for learning how data science can drive pricing strategies.
