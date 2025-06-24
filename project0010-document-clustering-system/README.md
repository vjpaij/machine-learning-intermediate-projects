### Description:

Document clustering is used to automatically group similar text documents without predefined labels. This helps in organizing, summarizing, or discovering hidden topics in a corpus. In this project, we build a system that vectorizes documents using TF-IDF and clusters them using K-Means, then visualizes the results.

- Converts documents to numerical vectors using TF-IDF
- Groups similar documents using K-Means clustering
- Reduces dimensions with PCA for easy 2D plotting
- Helps visually understand document themes or topics

## Document Clustering with K-Means (Scikit-Learn)

This example demonstrates how to cluster a set of short documents (texts) into groups based on their content using machine learning techniques. Here's a step-by-step explanation of the code and what the output means.

### Step-by-Step Code Explanation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
```

These are the required libraries:

* `TfidfVectorizer`: Converts text into numerical feature vectors based on TF-IDF (Term Frequency-Inverse Document Frequency).
* `KMeans`: An unsupervised machine learning algorithm used for clustering.
* `PCA`: Principal Component Analysis used to reduce high-dimensional data into 2D for visualization.
* `matplotlib.pyplot`: For plotting graphs.

```python
documents = [
    "Apple released a new iPhone today.",
    "The stock market saw major gains.",
    "Google announced a new Android update.",
    "Investors are optimistic about tech stocks.",
    "iPhones are selling fast this year.",
    "The economy is recovering from the pandemic.",
    "Samsung's Galaxy phones compete with iPhones.",
    "Financial experts predict more growth in the market.",
    "Android phones have improved camera features.",
    "Inflation rates remain a concern for investors."
]
```

These are the sample texts or documents that will be clustered. Some documents are about technology and phones, while others are related to finance and economics.

---

### Step 1: TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
```

* Converts the raw text into a matrix of TF-IDF features.
* `stop_words='english'` removes common English words (e.g., "the", "is") that don't add much meaning.
* The result `X` is a sparse matrix where each row represents a document, and each column represents a word/term.

---

### Step 2: K-Means Clustering

```python
n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X)
labels = model.labels_
```

* We choose `n_clusters = 3`, i.e., we want to group documents into 3 clusters.
* `KMeans` tries to find 3 clusters by minimizing the distance between points in the same cluster.
* `labels` is an array that assigns each document to a cluster (e.g., 0, 1, or 2).

---

### Step 3: Dimensionality Reduction with PCA

```python
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())
```

* TF-IDF vectors have high dimensions (equal to the number of unique words), so we use PCA to reduce the data to 2 dimensions for visualization.
* `X_reduced` now contains 2D points corresponding to each document.

---

### Step 4: Visualization

```python
plt.figure(figsize=(8, 5))
for i in range(n_clusters):
    plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=f"Cluster {i+1}")
plt.title("Document Clustering with K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots the documents as points in 2D space.
* Colors and labels the points by their cluster assignment.
* You’ll visually see 3 groups (or more if you set more clusters).

---

### Interpretation of Result

* Each document is assigned to a cluster based on its content.
* Documents in the same cluster are similar in terms of TF-IDF features (word importance).
* For example:

  * One cluster might group iPhone and Android related news.
  * Another cluster might include finance or market-related content.
  * A third cluster might include general economic content.

---

### Score/Prediction

* The model doesn't provide a classification accuracy because this is **unsupervised learning** — there are no true labels.
* You can use `.predict(new_texts)` on new documents (after vectorizing) to assign them to one of the learned clusters.
* To evaluate how well the clustering works, use metrics like silhouette score or manually inspect the clusters.

---

### Conclusion

This project is a simple but powerful demonstration of using:

* **TF-IDF** to convert text to vectors.
* **KMeans** for unsupervised document clustering.
* **PCA and matplotlib** for visualizing clusters.

It can be a foundation for document classification, topic modeling, or recommender systems.
