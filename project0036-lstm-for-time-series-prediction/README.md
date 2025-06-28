### Description:

An LSTM (Long Short-Term Memory) model is ideal for forecasting time series data where temporal dependencies matter. In this project, we use an LSTM-based neural network built with Keras to predict future values of a univariate time series (e.g., stock price, temperature).

- Builds an LSTM model for time series forecasting
- Handles sequence generation and reshaping for LSTM input
- Evaluates and visualizes predictions against actual values

# LSTM Time Series Prediction using PyTorch

This project demonstrates how to use an LSTM (Long Short-Term Memory) neural network in PyTorch for time series forecasting. It involves generating synthetic data, preparing it for LSTM training, training the model, and visualizing the predictions.

---

## Code Explanation

### 1. **Data Generation**

```python
np.random.seed(42)
time_steps = 200
data = np.sin(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.2, time_steps)
```

* Generates a sine wave with added Gaussian noise to simulate a real-world time series.

### 2. **Data Preparation**

```python
df = pd.DataFrame({'Value': data})
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
```

* The sine wave is stored in a DataFrame and normalized to the range \[0, 1] using `MinMaxScaler` to help neural network training.

### 3. **Creating Sequences**

```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
```

* Creates sliding window sequences. For each time step, `window_size` previous values are input (`X`), and the next value is the target (`y`).

### 4. **Train-Test Split**

```python
split = int(0.8 * len(X))
```

* Splits 80% for training and 20% for testing.

### 5. **Tensor Conversion**

```python
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
```

* Converts NumPy arrays to PyTorch tensors. The `unsqueeze(-1)` adds a channel dimension required by LSTM input format: `(batch_size, sequence_length, input_size)`.
In this case unsquuze isn't required as X_train is already shaped like (samples, window_size, 1)

### 6. **DataLoader for Batching**

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

* Enables efficient mini-batch training.

### 7. **Model Definition**

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        ...
```

* An LSTM model followed by a Linear layer. Only the final output of the LSTM is used for prediction.

### 8. **Training the Model**

```python
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        ...
```

* Trains for 20 epochs using MSELoss and Adam optimizer.
* Each batch computes loss, performs backpropagation, and updates weights.

### 9. **Prediction and Inverse Scaling**

```python
y_pred = model(X_test).squeeze().numpy()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
```

* Predicts on test data and rescales the predictions back to the original value range.

### 10. **Plotting the Results**

```python
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
```

* Compares the true values vs predicted values visually.

---

## Result & Interpretation

* **Plot**: Shows the LSTM model's ability to learn the underlying pattern in the noisy sine wave.
* **Prediction**: The dashed line follows the solid line closely, indicating good forecasting performance.
* **Loss**: Though not printed, the model uses MSE (Mean Squared Error) loss, which measures average squared difference between actual and predicted values. Lower is better.

---

## Summary

| Component     | Description                                  |
| ------------- | -------------------------------------------- |
| Data          | Noisy sine wave (synthetic time series)      |
| Model         | LSTM + Linear layer                          |
| Input Shape   | `(batch_size, window_size=10, input_size=1)` |
| Output        | One-step-ahead prediction                    |
| Loss          | Mean Squared Error (MSE)                     |
| Optimizer     | Adam                                         |
| Visualization | Actual vs Predicted plot                     |

This simple project shows how an LSTM can effectively forecast future values in a time series. It can be adapted to real-world datasets like stock prices, temperature readings, etc.
