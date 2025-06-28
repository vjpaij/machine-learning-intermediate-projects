### Description:

A GRU (Gated Recurrent Unit) network is a simplified and efficient variant of LSTM, ideal for modeling time-dependent sequences. In this project, we implement a GRU-based model to predict the next value in a univariate time series.

- Implements a GRU network for time series prediction
- Handles sequence windowing and reshaping for RNN input
- Visualizes model accuracy by comparing predicted vs actual output

## GRU Time Series Prediction with PyTorch

### Overview

This project demonstrates a simple time series forecasting pipeline using a Gated Recurrent Unit (GRU) neural network implemented in PyTorch. It simulates and predicts sequential data (cosine wave + noise) using a deep learning model.

---

### 1. **Data Generation and Preprocessing**

```python
np.random.seed(42)
time_steps = 200
data = np.cos(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.2, time_steps)
```

* A cosine wave is generated with added Gaussian noise to simulate real-world time series data.
* Total points = 200.

```python
df = pd.DataFrame({'Value': data})
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
```

* Data is normalized to range \[0,1] using `MinMaxScaler`, which helps neural networks converge faster.

### 2. **Sequence Creation**

```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
```

* This function converts the time series into input-output pairs for supervised learning.
* Each input `X` is a window of size 10, and the output `y` is the next value.

### 3. **Train-Test Split and Tensor Conversion**

```python
split = int(len(X) * 0.8)
```

* 80% for training, 20% for testing.
* Data is converted to PyTorch tensors and reshaped to fit GRU expectations: `(batch_size, sequence_length, features)`.

### 4. **Model Architecture**

```python
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)
```

* **GRU Layer**: Captures temporal dependencies.
* **Linear Layer**: Maps GRU output to final prediction.
* **Last time step** output is used for prediction.

### 5. **Training Loop**

```python
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x).squeeze()
        loss = criterion(output, batch_y.squeeze())
        loss.backward()
        optimizer.step()
```

* Mean Squared Error (MSE) is used as the loss function.
* Adam optimizer updates the model parameters.

### 6. **Evaluation & Plotting**

```python
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()
```

* Model is evaluated on test data without computing gradients.

```python
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
```

* Predictions and actuals are transformed back to the original scale.

### 7. **Results/Plot**

```python
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
```

* The plot visually compares actual vs predicted values.
* A close match indicates good learning of patterns.

---

### What Does the Result Mean?

* The GRU model learns temporal dependencies from historical windows and forecasts future values.
* The plotted lines (Actual vs Predicted) indicate how well the model captures the time series trend.
* The less deviation between the curves, the better the model performance.

---

### Improvements to Consider

* Increase data points for robustness.
* Try hyperparameter tuning.
* Experiment with deeper GRUs, LSTMs, or attention mechanisms.
* Introduce more real-world complexity like multivariate inputs.
