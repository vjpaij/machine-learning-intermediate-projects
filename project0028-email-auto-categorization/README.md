### Description:

Email auto-categorization automatically classifies emails into predefined folders like "Work", "Promotions", "Personal", or "Spam". This project builds a multi-class classifier using TF-IDF features and a Logistic Regression model trained on the subject and body of simulated emails.

- Transforms raw email content into TF-IDF vectors
- Trains a multi-class classifier to categorize messages
- Uses Logistic Regression with text data
- Visualizes model performance with a confusion matrix

## Email Classification using Logistic Regression

This project demonstrates a simple email categorization system using machine learning. The goal is to classify emails into different categories such as Work, Promotions, Personal, and Spam based on their content.

### Code Explanation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

These are the required libraries:

* `pandas` for data manipulation.
* `scikit-learn` for ML pipeline (splitting data, vectorizing text, training and evaluating model).
* `matplotlib` and `seaborn` for visualizing the confusion matrix.

```python
# Simulated email dataset
data = {
    'EmailText': [...],
    'Category': [...]
}
df = pd.DataFrame(data)
```

A sample dataset is created containing `EmailText` and corresponding `Category` labels. Categories include: Work, Promotions, Personal, Spam.

```python
# Encode labels
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
```

* Converts string labels into numeric labels (e.g., Work=3, Spam=2, etc.).
* `label_map` stores the mapping from numeric code back to string label.

```python
# Vectorize email text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['EmailText'])
y = df['Category_Label']
```

* Text data is vectorized using TF-IDF, which converts words into numerical features while down-weighting common words (stop words).
* `X` contains the TF-IDF features, `y` contains the encoded labels.

```python
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* The dataset is split into training (70%) and test (30%) sets.

```python
# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

* A Logistic Regression model is trained on the training data.
* `max_iter=1000` ensures convergence.

```python
# Predict and evaluate
y_pred = model.predict(X_test)
```

* Model makes predictions on the test set.

```python
print(classification_report(y_test, y_pred, labels=label, target_names=label_map.values()))
```

* Prints precision, recall, F1-score, and support for each category.

```python
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=label)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Confusion Matrix - Email Categorization")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png")
```

* Displays a heatmap showing actual vs predicted categories.
* Each cell in the matrix represents how often an actual class was predicted as a different class.

### Output and Interpretation

**Classification Report:**

* **Precision**: Out of all predicted instances of a class, how many were correct?
* **Recall**: Out of all actual instances of a class, how many were correctly predicted?
* **F1-score**: Harmonic mean of precision and recall.

**Confusion Matrix:**

* Diagonal cells indicate correct predictions.
* Off-diagonal cells indicate misclassifications.
* Helps to visually assess which categories are being confused.

### Summary

* This code demonstrates a basic email classifier.
* Using TF-IDF and Logistic Regression, it predicts whether an email belongs to Work, Promotions, Personal, or Spam.
* The model is evaluated using classification metrics and visualized with a confusion matrix.

This is a foundational approach suitable for understanding text classification pipelines.
