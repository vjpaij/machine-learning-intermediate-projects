import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Generate synthetic time series data
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=100, freq='D')
values = np.random.normal(loc=100, scale=5, size=100)
values[20] += 30   # Injected anomaly
values[75] -= 25   # Injected anomaly
 
# Create DataFrame
df = pd.DataFrame({'Date': time, 'Value': values})
df.set_index('Date', inplace=True)
 
# Rolling window for mean and standard deviation
window_size = 10
df['RollingMean'] = df['Value'].rolling(window=window_size).mean()
df['RollingStd'] = df['Value'].rolling(window=window_size).std()
 
# Define anomaly threshold (mean ± 2*std)
df['Anomaly'] = ((df['Value'] > df['RollingMean'] + 2*df['RollingStd']) |
                 (df['Value'] < df['RollingMean'] - 2*df['RollingStd']))
 
# Plot the time series with anomalies
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
plt.savefig('anomaly_detection_time_series.png')