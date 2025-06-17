### Description:

Fraud detection systems are designed to identify unusual behavior in financial transactions that may indicate fraud. In this project, we implement a binary classifier using logistic regression on an imbalanced dataset (simulated) to predict fraudulent transactions. We also apply resampling techniques to handle class imbalance and evaluate using appropriate metrics.

- Detects rare events (fraud) using logistic regression
- Handles imbalanced data via oversampling
- Evaluates using precision, recall, ROC-AUC, and confusion matrix

## Fraud Detection with Logistic Regression

This project demonstrates a basic fraud detection system using logistic regression. It includes data simulation, handling class imbalance, model training, and evaluation.

### Code Explanation

#### 1. **Importing Libraries**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
```

We import necessary libraries for data manipulation (`numpy`, `pandas`), machine learning (`sklearn`), and visualization (`seaborn`, `matplotlib`).

#### 2. **Data Simulation**

```python
np.random.seed(42)
n_samples = 1000
fraud_ratio = 0.05
```

A reproducible random seed is set, with 1000 total samples and 5% labeled as fraud.

```python
data = {
    'TransactionAmount': np.random.exponential(scale=100, size=n_samples),
    'TransactionTime': np.random.randint(0, 24, size=n_samples),
    'IsForeign': np.random.randint(0, 2, size=n_samples),
    'IsHighRiskCountry': np.random.randint(0, 2, size=n_samples),
    'Fraud': np.zeros(n_samples, dtype=int)
}
```

We generate features:

* `TransactionAmount`: follows an exponential distribution.
* `TransactionTime`: hour of transaction.
* `IsForeign` & `IsHighRiskCountry`: binary flags.
* `Fraud`: initially set to 0.

```python
fraud_indices = np.random.choice(n_samples, int(fraud_ratio * n_samples), replace=False)
for idx in fraud_indices:
    data['Fraud'][idx] = 1
```

Randomly assign 5% of the transactions as fraud.

#### 3. **Data Preparation**

```python
df = pd.DataFrame(data)
```

Convert dictionary to DataFrame.

```python
df_majority = df[df.Fraud == 0]
df_minority = df[df.Fraud == 1]
```

Split data into majority (non-fraud) and minority (fraud) classes.

```python
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
```

Perform upsampling to balance the dataset, which is critical for handling class imbalance in fraud detection.

#### 4. **Model Training**

```python
X = df_balanced.drop('Fraud', axis=1)
y = df_balanced['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Separate features and target, then split into training and testing sets.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

Train a logistic regression classifier.

#### 5. **Model Evaluation**

```python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

Predict class labels and probabilities.

```python
print(classification_report(y_test, y_pred))
```

Outputs metrics like precision, recall, F1-score, and support.

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix - Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

Visualizes how well the classifier distinguishes between fraud and non-fraud cases.

```python
roc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_score:.2f}")
```

Calculates the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), a robust metric for classification performance.

### Interpretation of Results

* **Classification Report**: Includes precision, recall, F1-score.

  * High **precision** for fraud class means few false positives.
  * High **recall** for fraud class means most frauds are detected.
* **Confusion Matrix**: Shows true/false positives/negatives.

  * True Positives (Fraud correctly detected)
  * False Negatives (Fraud missed)
  * False Positives (Normal marked as fraud)
  * True Negatives (Normal correctly identified)
* **ROC-AUC Score**: Ranges from 0.5 (random) to 1.0 (perfect).

  * Closer to 1 means better model performance.

### Conclusion

This project showcases how to build a simple but effective fraud detection system with balanced data, logistic regression, and evaluation via classification metrics and ROC-AUC. While this is a simulation, it serves as a solid baseline for real-world implementations.
