### Description:

Sentiment analysis determines whether text expresses positive, negative, or neutral emotion. In this project, we build a binary sentiment classifier for movie reviews using the IMDb dataset (or custom input). We’ll apply TF-IDF vectorization and train a Logistic Regression model using scikit-learn.

- Uses TF-IDF vectorization for converting text into numeric form
- Trains a logistic regression classifier for binary sentiment detection
- Visualizes results using confusion matrix and detailed performance report

## Sentiment Analysis with Logistic Regression

This script demonstrates a simple sentiment analysis pipeline using scikit-learn, ideal for educational and prototype purposes.

### 🛠 Installation

Make sure you have the required libraries installed:

```bash
pip install sklearn matplotlib seaborn
```

### 📄 Description

This code uses a small set of movie reviews labeled as either positive (`1`) or negative (`0`) and trains a logistic regression classifier to predict sentiment.

### 🔄 Workflow Breakdown

1. **Import Libraries**

```python
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

These libraries handle text preprocessing, model training, evaluation, and plotting.

2. **Prepare Dataset**

```python
reviews = [
    "I absolutely loved this movie. The story was fantastic and acting was brilliant.",
    "This was a waste of time. Horrible plot and bad direction.",
    "Great film! Will definitely watch again.",
    "Terrible experience. I walked out halfway.",
    "Not bad, but could have been better.",
    "Best movie I’ve seen this year!",
    "Awful. Just awful. Please don’t watch.",
    "A masterpiece. Well done to the cast.",
    "I didn’t like the movie. It was boring.",
    "Amazing movie! Beautiful visuals and emotional story."
]
labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
```

We use a simple predefined dataset of 10 reviews and their sentiment labels.

3. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42)
```

This splits the dataset into 70% training and 30% testing data.

4. **Text Vectorization with TF-IDF**

```python
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

TF-IDF is used to convert text data into numerical vectors while removing common English stopwords.

5. **Train the Model**

```python
model = LogisticRegression()
model.fit(X_train_vec, y_train)
```

We train a logistic regression model using the vectorized training data.

6. **Evaluate the Model**

```python
y_pred = model.predict(X_test_vec)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
```

We print out precision, recall, and F1-score using the classification report.

7. **Confusion Matrix Visualization**

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.title("Confusion Matrix - Sentiment Analysis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

This block visualizes the confusion matrix to better understand the classification performance.

### ✅ Output

You will see a classification report and a confusion matrix heatmap showing how well the model performed on the test data.

### 📌 Notes

* For production use, use a larger dataset and consider deep learning models or more advanced feature extraction methods.
* Tune hyperparameters (e.g., vectorizer options or logistic regression settings) for improved accuracy.

---

Happy Sentiment Analyzing! 🚀
