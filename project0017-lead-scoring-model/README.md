### Description:

A lead scoring model predicts how likely a sales lead is to convert into a paying customer. This helps prioritize leads and improve marketing efficiency. In this project, we build a classification model to score leads using features like demographics, behavior, and engagement level.

- Builds a lead scoring classifier using marketing and behavioral features
- Handles categorical encoding and scaling
- Uses a Random Forest for robust prediction and interpretability
- Evaluates model with classification report and confusion matrix

### Lead Conversion Prediction Model (README)

This script builds a simple machine learning pipeline to predict whether a lead will convert or not based on user behavior and lead source.

---

## ✨ Project Overview

This notebook demonstrates how to:

1. Preprocess and encode categorical and numeric features
2. Split the dataset into training and test sets
3. Train a Random Forest classification model
4. Evaluate the model using classification metrics and a confusion matrix

---

## 📈 Dataset Details

The dataset is a simulated collection of leads with the following features:

| Column     | Description                          |
| ---------- | ------------------------------------ |
| LeadSource | Origin of the lead (Email, Ad, etc.) |
| Age        | Age of the lead                      |
| PageViews  | Pages viewed on the website          |
| TimeOnSite | Time spent on site (in seconds)      |
| Converted  | Target: 1 if converted, 0 otherwise  |

---

## 💡 Code Explanation (Step-by-Step)

```python
# Step 1: Data setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

**Reasoning**: Imports essential libraries for data handling, visualization, machine learning, and evaluation.

```python
data = { ... }
df = pd.DataFrame(data)
```

Creates a DataFrame with 10 rows of sample lead data.

```python
# Step 2: Encode 'LeadSource' as numeric
le = LabelEncoder()
df['LeadSource'] = le.fit_transform(df['LeadSource'])
```

**Reasoning**: Machine learning models work only with numeric data. Label encoding converts categories like "Email" to integers.

```python
# Step 3: Define features and target
X = df.drop('Converted', axis=1)
y = df['Converted']
```

Separates the dataset into independent variables `X` and dependent variable `y`.

```python
# Step 4: Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Reasoning**: StandardScaler brings features to a common scale, improving model performance.

```python
# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

**Reasoning**: 70% of the data is used for training, 30% for testing. `random_state` ensures reproducibility.

```python
# Step 6: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Reasoning**: Uses a Random Forest, an ensemble method that builds multiple decision trees and averages results.

```python
# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Generates a prediction and prints a **classification report** with:

* **Precision**: How many predicted positive leads were actually positive?
* **Recall**: How many actual positive leads were predicted correctly?
* **F1-Score**: Harmonic mean of precision and recall.
* **Support**: Number of samples per class.

```python
# Step 8: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ...)
plt.show()
```

**Confusion Matrix**: Visual breakdown of predictions:

* True Positives (TP): Correctly predicted conversions
* True Negatives (TN): Correctly predicted non-conversions
* False Positives (FP): Predicted converted but didn’t
* False Negatives (FN): Missed actual conversions

---

## 📊 Sample Output

```
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         2

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3
```

**Interpretation**:

* Perfect precision, recall, and f1-score in this example due to small size and possibly easy separability.
* Accuracy: 100% — All predictions matched actual labels (but note: with very small test sets, this may not generalize).

---

## 📆 Improvements for Real-World Use

* Use larger, real-world datasets
* Perform cross-validation
* Tune hyperparameters with GridSearchCV
* Handle class imbalance if present
* Engineer additional features

---

## 🚀 Conclusion

This script shows a full pipeline for predicting lead conversion using a simple Random Forest classifier. It is a solid base for more complex models in marketing, sales, and CRM analytics.

---


