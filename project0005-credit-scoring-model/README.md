### Description:

A credit scoring model predicts the likelihood that a borrower will default on a loan. It’s widely used by banks and lenders to assess credit risk. In this project, we build a basic binary classification model using logistic regression on simulated applicant data to predict whether an applicant is likely to default.

- Builds a credit risk classifier using applicant data
- Handles categorical encoding and feature scaling
- Evaluates results using a classification report and confusion matrix

## Credit Scoring Model using Logistic Regression

This script demonstrates a simple credit scoring model built using logistic regression. It processes a synthetic dataset of credit applicants to predict whether an individual will default on a loan. Below is a detailed explanation of each part of the code:

### 1. **Importing Libraries**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

These libraries are essential for data manipulation (`pandas`), preprocessing (`StandardScaler`, `LabelEncoder`), model building (`LogisticRegression`), evaluation (`classification_report`, `confusion_matrix`), and visualization (`seaborn`, `matplotlib`).

### 2. **Creating the Dataset**

```python
data = {
    'Age': [...],
    'Income': [...],
    'LoanAmount': [...],
    'CreditHistory': [...],
    'EmploymentStatus': [...],
    'Default': [...]
}
df = pd.DataFrame(data)
```

A synthetic dataset is created with applicant information. The target variable is `Default`, indicating whether a person defaulted on their loan.

### 3. **Encoding Categorical Variables**

```python
le = LabelEncoder()
df['CreditHistory'] = le.fit_transform(df['CreditHistory'])
df['EmploymentStatus'] = le.fit_transform(df['EmploymentStatus'])
df['Default'] = le.fit_transform(df['Default'])
```

Categorical variables are converted into numeric form using `LabelEncoder` so they can be used in machine learning models.

* 'CreditHistory': Good=1, Bad=0
* 'EmploymentStatus': Encoded to integers
* 'Default': No=0, Yes=1

### 4. **Defining Features and Target Variable**

```python
X = df.drop('Default', axis=1)
y = df['Default']
```

The feature matrix `X` contains the predictors, and the target vector `y` contains the labels (default status).

### 5. **Feature Scaling**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization is performed to bring all features onto the same scale, which improves model performance.

### 6. **Splitting the Dataset**

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

The dataset is split into training (70%) and testing (30%) subsets to evaluate model performance.

### 7. **Training the Logistic Regression Model**

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

A logistic regression model is trained on the training data.

### 8. **Model Prediction and Evaluation**

```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

The model's predictions are compared to the actual values using classification metrics such as precision, recall, F1-score, and accuracy.

### 9. **Visualizing the Confusion Matrix**

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title("Confusion Matrix - Credit Scoring Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

A heatmap of the confusion matrix visually displays how many correct and incorrect predictions the model made.

---

This script is a foundational example of using logistic regression for binary classification problems in credit scoring. It shows the full pipeline: data preparation, model training, evaluation, and visualization.
