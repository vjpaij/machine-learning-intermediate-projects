### Description:

Sentiment analysis for social media detects emotions and opinions (positive, negative, neutral) from user-generated content like tweets or comments. This project builds a sentiment classifier using TF-IDF and Logistic Regression on simulated social media posts.

- Applies sentiment analysis to classify social media posts
- Uses TF-IDF for feature extraction and Logistic Regression for classification
- Evaluates with precision, recall, and confusion matrix

# Sentiment Analysis on Social Media Posts

This project demonstrates a simple yet effective sentiment analysis pipeline using logistic regression on a simulated social media dataset. It classifies posts into **Positive**, **Negative**, or **Neutral** sentiments.

---

## 📦 Dataset

We create a synthetic dataset of 10 social media posts with manually labeled sentiments:

```python
Post: A short social media message or review
Sentiment: Label ('Positive', 'Negative', or 'Neutral')
```

### Sample Data

| Post                                              | Sentiment |
| ------------------------------------------------- | --------- |
| I love the new features on this app! So smooth!   | Positive  |
| Worst update ever. Totally ruined the experience. | Negative  |
| Neutral about the update, nothing special.        | Neutral   |

---

## ⚙️ Preprocessing Steps

### 1. **Label Encoding**

The sentiment labels are encoded as numeric classes using pandas category codes:

```python
Positive = 2
Negative = 0
Neutral  = 1
```

### 2. **Text Vectorization (TF-IDF)**

We use `TfidfVectorizer` to convert the text data into numerical features by:

* Removing English stopwords
* Capturing the importance of words using Term Frequency-Inverse Document Frequency (TF-IDF)

---

## 🧠 Model Training

A **Logistic Regression** classifier is trained using the vectorized features.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

The data is split 70:30 into training and testing sets using `train_test_split` with a fixed `random_state=42` for reproducibility.

---

## 📊 Evaluation

### 1. **Classification Report**

The classification report includes:

* **Precision**: Correct positive predictions / Total predicted as positive
* **Recall**: Correct positive predictions / Total actual positives
* **F1-score**: Harmonic mean of precision and recall
* **Support**: Number of actual instances per class

Example Output:

```
              precision    recall  f1-score   support

    Negative       1.00      1.00      1.00         1
     Neutral       0.00      0.00      0.00         1
    Positive       1.00      1.00      1.00         1
```

Note: With such a small dataset (3 test samples), results can vary significantly. Here, the Neutral post was misclassified, leading to a 0 score.

### 2. **Confusion Matrix**

A heatmap displays the confusion matrix, which shows:

* Rows = Actual sentiment
* Columns = Predicted sentiment

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
```

This helps visualize where misclassifications occurred.

---

## 🔍 Interpretation

The logistic regression model makes predictions based on the TF-IDF feature importance. While the classifier shows good accuracy on Positive and Negative classes, it misclassifies Neutral posts due to the limited data and overlapping language.

---

## ✅ Summary

* Simple text classification using logistic regression
* Preprocessing using TF-IDF
* Evaluation using classification report & confusion matrix
* Good baseline but limited by data size

This script can be extended with more data and advanced models for better performance.

---
