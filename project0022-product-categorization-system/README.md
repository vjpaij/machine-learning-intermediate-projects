### Description:

A product categorization system automatically classifies products into categories based on their name or description. This improves product discovery, search relevance, and catalog management. In this project, we use TF-IDF vectorization with a Logistic Regression classifier to predict product categories.

- Turns product names into TF-IDF vectors
- Trains a text classifier to predict product categories
- Evaluates with classification report and confusion matrix

# Product Categorization using TF-IDF and Logistic Regression

This project demonstrates a basic text classification task: categorizing products based on their names using machine learning. The process includes feature extraction using TF-IDF, model training with logistic regression, and evaluation with a classification report and confusion matrix.

## Code Explanation and Reasoning

### 1. Importing Libraries

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

* **pandas**: Used for data manipulation and analysis.
* **TfidfVectorizer**: Converts text to numeric vectors using Term Frequency-Inverse Document Frequency.
* **LogisticRegression**: A simple yet effective classification algorithm.
* **train\_test\_split**: Splits data into training and test sets.
* **classification\_report & confusion\_matrix**: For evaluating model performance.
* **seaborn & matplotlib**: For visualization.

### 2. Simulated Product Data

```python
data = {
    'ProductName': [...],
    'Category': [...]
}
df = pd.DataFrame(data)
```

* A sample dataset is created with product names and their associated categories like Electronics, Furniture, etc.

### 3. Label Encoding

```python
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
```

* Converts string labels (e.g., "Electronics") into numeric labels (e.g., 0, 1, 2) for model compatibility.
* `label_map` helps in translating numeric predictions back to readable category names.

### 4. Feature Extraction using TF-IDF

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ProductName'])
y = df['Category_Label']
```

* TF-IDF transforms each product name into a vector based on term importance.
* `X` is the feature matrix; `y` is the target label vector.

### 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* 70% of the data is used for training; 30% for testing.

### 6. Model Training

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

* A logistic regression model is trained with the feature matrix `X_train` and labels `y_train`.
* `max_iter=1000` ensures convergence.

### 7. Prediction and Evaluation

```python
y_pred = model.predict(X_test)
```

* The model predicts categories for unseen test data (`X_test`).

#### Classification Report

```python
labels = list(label_map.keys())
print(classification_report(y_test, y_pred, labels = label, target_names=label_map.values()))
```

* Displays precision, recall, f1-score, and support for each category:

  * **Precision**: Out of all predicted instances of a class, how many were actually correct?
  * **Recall**: Out of all actual instances of a class, how many were correctly predicted?
  * **F1-score**: Harmonic mean of precision and recall.
  * **Support**: Number of actual instances for each class in the test set.

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred, labels = label)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```

* Visualizes the performance of the classifier in predicting each class.
* Diagonal values indicate correct predictions; off-diagonal values indicate misclassifications.

## Result Interpretation

The printed classification report and confusion matrix give insight into how well the model performs. In a real-world scenario, higher support per class and more data would yield a more reliable evaluation. Here, due to the small dataset:

* The model may overfit to the training data.
* Performance metrics might fluctuate significantly across runs.

Despite this, the workflow demonstrates the fundamental steps of:

* Preprocessing textual data
* Converting it into machine-readable format
* Training a classifier
* Evaluating its performance using well-established metrics and visual tools

This pipeline can be expanded and applied to larger datasets and more complex text classification tasks in e-commerce, customer support, or document management.

---

**Note**: For production use, consider:

* More training data
* Hyperparameter tuning
* Better feature engineering
* Advanced models like SVM, Random Forest, or BERT-based transformers
