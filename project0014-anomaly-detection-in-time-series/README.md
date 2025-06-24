### Description:

Anomaly detection in time series identifies unusual patterns, spikes, or dips that deviate from normal behavior. This is useful in monitoring systems, fraud detection, and predictive maintenance. In this project, we use a statistical method (rolling mean and standard deviation) to detect anomalies in synthetic time series data.

- Uses rolling statistics to define a normal behavior range
- Detects outliers based on deviations beyond a defined threshold
- Highlights visual anomalies in a time series plot

## Time Series Anomaly Detection Using Rolling Statistics

### Overview

This script demonstrates a basic approach to detecting anomalies in time series data using rolling statistics (mean and standard deviation). It simulates synthetic data, calculates moving averages and thresholds, identifies anomalies, and visualizes the results.

---

### Code Walkthrough with Explanation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

These imports bring in necessary libraries:

* `numpy`: for generating random numbers and numerical operations.
* `pandas`: for handling date/time series and data manipulation.
* `matplotlib.pyplot`: for plotting the results.

```python
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=100, freq='D')
values = np.random.normal(loc=100, scale=5, size=100)
values[20] += 30   # Injected anomaly
values[75] -= 25   # Injected anomaly
```

* We set a random seed for reproducibility.
* `time` is a 100-day daily date range starting from Jan 1, 2023.
* `values` are normally distributed around 100 with a standard deviation of 5.
* Two anomalies are manually injected: a spike at index 20 and a dip at index 75.

```python
df = pd.DataFrame({'Date': time, 'Value': values})
df.set_index('Date', inplace=True)
```

* A DataFrame `df` is created with the date as the index and the synthetic values.

```python
window_size = 10
df['RollingMean'] = df['Value'].rolling(window=window_size).mean()
df['RollingStd'] = df['Value'].rolling(window=window_size).std()
```

* A rolling window of size 10 is used to calculate the moving average and standard deviation.
* This allows the model to understand the local behavior of the data.

```python
df['Anomaly'] = ((df['Value'] > df['RollingMean'] + 2*df['RollingStd']) |
                 (df['Value'] < df['RollingMean'] - 2*df['RollingStd']))
```

* An anomaly is flagged when a value lies outside the threshold: **mean ± 2 \* std**.
* This method assumes a normal distribution (95% of data lies within 2 standard deviations).

```python
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Value'], label='Value')
plt.plot(df.index, df['RollingMean'], color='orange', label='Rolling Mean')
plt.fill_between(df.index,
                 df['RollingMean'] + 2*df['RollingStd'],
                 df['RollingMean'] - 2*df['RollingStd'],
                 color='lightgray', alpha=0.5, label='Threshold Range')
plt.scatter(df[df['Anomaly']].index, df[df['Anomaly']]['Value'], color='red', label='Anomalies', zorder=5)
plt.title("Anomaly Detection in Time Series")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* This block generates a plot:

  * Blue line = actual values
  * Orange line = rolling mean
  * Gray band = acceptable range (mean ± 2\*std)
  * Red dots = detected anomalies

---

### Output and Interpretation

The output is a time series plot that clearly highlights:

* Normal values lying within the gray band.
* Two anomalies: one positive spike and one negative dip, which exceed the 2 standard deviation range.

#### What the Result Means:

* If a value is **significantly different** from the recent average, it is considered an anomaly.
* This is useful for detecting **unexpected events**, such as:

  * Sensor malfunctions
  * Fraudulent transactions
  * Stock price spikes/drops

---

### Limitations

* Assumes data is normally distributed.
* Only works well with **stationary time series**.
* More complex methods like Isolation Forests, Autoencoders, or Prophet may be needed for real-world, non-linear data.

---

### Summary

This script provides a simple, explainable way to detect anomalies in time series using moving averages and standard deviation thresholds. It’s a great starting point for understanding time series anomaly detection.

---

