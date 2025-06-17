### Description:

Customer churn prediction aims to identify customers likely to leave (churn) a service or subscription. By analyzing customer behavior, businesses can proactively retain at-risk users. In this project, we use a logistic regression model to predict churn from customer data using scikit-learn.

- Predicts churn (Yes/No) using behavioral and contract features
- Preprocesses both categorical and numerical data
- Evaluates performance with classification metrics and confusion matrix

### Customer Churn Prediction using Logistic Regression

This Python script demonstrates a basic machine learning pipeline for predicting customer churn using a logistic regression model. The code uses simulated customer data and includes preprocessing, model training, evaluation, and visualization steps.

#### 1. **Importing Libraries**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

* `pandas`: For data manipulation.
* `sklearn`: For preprocessing, model building, and evaluation.
* `seaborn` and `matplotlib`: For plotting the confusion matrix.

#### 2. **Simulated Data**

```python
data = {
    'Age': [25, 45, 39, 50, 30, 28, 55, 23, 40, 35],
    'MonthlyCharges': [50, 80, 70, 100, 60, 55, 90, 45, 75, 65],
    'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Two year',
                 'Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
    'Tenure': [1, 24, 5, 18, 36, 3, 20, 2, 28, 12],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)
```

* Creates a DataFrame with customer demographic and subscription details.

#### 3. **Encoding Categorical Variables**

```python
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['Churn'] = le.fit_transform(df['Churn'])
```

* Converts categorical variables into numerical values:

  * Contract: 'Month-to-month' = 0, 'One year' = 1, 'Two year' = 2
  * Churn: 'Yes' = 1, 'No' = 0

#### 4. **Feature and Target Split**

```python
X = df.drop('Churn', axis=1)
y = df['Churn']
```

* `X` includes input features.
* `y` is the target variable indicating churn status.

#### 5. **Feature Scaling**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* Standardizes features to have a mean of 0 and standard deviation of 1.

#### 6. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

* Splits the data into 70% training and 30% testing sets.

#### 7. **Model Training**

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

* Trains a logistic regression model on the training data.

#### 8. **Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
```

* Makes predictions on the test set.
* Prints precision, recall, f1-score, and accuracy.

#### 9. **Confusion Matrix Visualization**

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix - Customer Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

* Creates a heatmap for the confusion matrix showing actual vs predicted churn values.

---

This code provides a complete example of how to build and evaluate a logistic regression model for binary classification problems using basic customer churn data.
