### Description:

Text classification for news articles involves assigning categories (e.g., politics, sports, tech) to news content automatically. In this project, we build a multi-class text classifier using TF-IDF vectorization and a Multinomial Naive Bayes model to classify simulated news headlines into categories.

- Uses TF-IDF to convert text into numerical features
- Trains a Multinomial Naive Bayes model for multi-class classification
- Evaluates performance with detailed metrics and confusion matrix

## News Category Classifier using Naive Bayes

This project demonstrates a basic **text classification** pipeline that predicts the category of news headlines (e.g., Politics, Sports, Technology) using the **Multinomial Naive Bayes** model and **TF-IDF vectorization**.

### 📊 Dataset

The dataset is a simulated collection of 10 news headlines labeled across 5 categories:

* Politics
* Sports
* Technology
* Finance
* Science

Example:

```
'Elections bring new political changes in the country' --> 'Politics'
```

### 🔧 Step-by-Step Code Explanation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

**Libraries Used:**

* `pandas`: For tabular data handling
* `sklearn`: For machine learning utilities (model, preprocessing, evaluation)
* `seaborn`, `matplotlib`: For data visualization

---

```python
data = { ... }
df = pd.DataFrame(data)
```

* Create a small DataFrame from a dictionary containing headlines and their categories.

---

```python
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
```

* **Encoding Labels**: Converts category names into numeric labels required for modeling.
* `label_map`: Stores a reverse lookup from label numbers to category names for interpretation.

---

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Headline'])
y = df['Category_Label']
```

* **TF-IDF Vectorization**: Transforms text into numerical features by considering term frequency and penalizing common words (`stop_words='english'`).
* `X` contains the feature vectors for headlines.
* `y` is the corresponding category label.

---

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* Splits data into training (70%) and testing (30%) sets. Random state ensures reproducibility.

---

```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

* **Training** the Naive Bayes classifier using training data.
* MultinomialNB is effective for discrete text features like TF-IDF.

---

```python
y_pred = model.predict(X_test)
label = list(label_map.keys())
print(classification_report(y_test, y_pred, labels=label, target_names=label_map.values()))
```

* Predicts the categories for test data.
* Displays precision, recall, f1-score, and support for each class.

#### 🧾 What does the classification report show?

* **Precision**: Out of predicted X category, how many were actually X.
* **Recall**: Out of actual X category, how many were predicted as X.
* **F1-score**: Harmonic mean of precision and recall.
* **Support**: Number of samples in each class.

Due to small data, some scores may be 0.0 (no examples or poor prediction).

---

```python
cm = confusion_matrix(y_test, y_pred, labels=label)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ...)
```

* **Confusion Matrix**: Visual representation of predictions vs actual values.
* Diagonal values represent correct predictions.
* Off-diagonal values represent misclassifications.

### 🧠 Interpretation

This project illustrates how simple machine learning models can classify text into categories. Despite the small dataset, it gives insights into how natural language processing (NLP) pipelines are built and evaluated.

To improve this model:

* Use a larger, real-world dataset.
* Try other models (e.g., SVM, Logistic Regression).
* Perform hyperparameter tuning.

---

### ✅ Summary

* Used Naive Bayes classifier for text classification.
* Applied TF-IDF to convert text to features.
* Evaluated model with classification report and confusion matrix.

This example forms a good starting point for news article classification and text analytics projects.

