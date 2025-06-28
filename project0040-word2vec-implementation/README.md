### Description:

Word2Vec is a neural network-based model that learns dense vector representations for words based on their context, capturing semantic relationships (e.g., "king - man + woman ≈ queen"). In this project, we implement Word2Vec using Gensim to train embeddings on a sample corpus and visualize word similarities.

- Trains Word2Vec embeddings using Gensim on real-world text
- Captures semantic similarity between words
- Visualizes relationships using PCA

## Word2Vec Embedding with NLTK Brown Corpus - Code Explanation and Report

### Overview

This script trains a Word2Vec model using the Brown corpus from NLTK, explores semantic similarities between words, and visualizes the word embeddings using PCA (Principal Component Analysis). The output includes the most similar words to a given word and a 2D plot showing semantic relationships among selected words.

---

### Detailed Code Explanation

```python
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```

* **Imports**:

  * `nltk`: Natural Language Toolkit for working with corpora.
  * `gensim.models.Word2Vec`: To train and use Word2Vec models.
  * `matplotlib.pyplot`: For visualization.
  * `PCA` from `sklearn`: To reduce high-dimensional vectors into 2D for visualization.

```python
nltk.download('punkt')
nltk.download('brown')
```

* Downloads required NLTK resources:

  * `punkt`: For tokenization (not directly used here since Brown is pre-tokenized).
  * `brown`: The Brown corpus, a large collection of pre-tokenized English text.

```python
sentences = brown.sents()  # Pre-tokenized corpus from NLTK
```

* Loads the Brown corpus as a list of tokenized sentences.

```python
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, sg=1, workers=4)
```

* Trains a Word2Vec model:

  * `vector_size=100`: Embedding dimension is 100.
  * `window=5`: Context window size is 5 words.
  * `min_count=5`: Ignores words with total frequency < 5.
  * `sg=1`: Uses Skip-gram (instead of CBOW).
  * `workers=4`: Uses 4 CPU threads.

```python
print("\nMost similar words to 'money':")
print(model.wv.most_similar('money'))
```

* Prints the top 10 words most similar to 'money' based on cosine similarity in embedding space.
* These reflect semantic similarity learned from the Brown corpus.

```python
words = ['king', 'queen', 'man', 'woman', 'money', 'bank', 'school', 'teacher', 'student', 'war']
word_vecs = [model.wv[word] for word in words]
```

* Selects a few example words.
* Extracts their 100-dimensional embeddings.

```python
pca = PCA(n_components=2)
result = pca.fit_transform(word_vecs)
```

* Reduces the 100D word vectors into 2D using PCA for visualization.

```python
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.text(result[i, 0] + 0.01, result[i, 1] + 0.01, word)
plt.title("Word2Vec Word Embeddings (PCA Projection)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

* Plots each word in a 2D space.
* Words with similar meanings or contexts should appear closer together in this plot.

---

### Report and Interpretation

#### 1. **Most Similar Words to 'money'**

The `most_similar('money')` output might look like:

```python
[('cash', 0.84), ('loan', 0.81), ('bank', 0.78), ('credit', 0.76), ...]
```

* Each tuple shows a word and its cosine similarity score.
* High similarity indicates similar context usage in the Brown corpus.
* For example, 'money' and 'bank' are often used in similar financial contexts.

#### 2. **PCA Word Embedding Plot**

* The PCA plot shows semantic clustering:

  * Words like 'king' and 'queen' may cluster together.
  * 'teacher', 'school', and 'student' might be in close proximity.
  * If 'war' appears far from 'bank', it reflects different semantic domains.
* This visual insight helps understand how Word2Vec captures meaning based on word co-occurrence.

---

### What Does It All Mean?

* **Word2Vec** turns words into vectors based on context.
* **Closer vectors = similar meanings**.
* This is foundational in many NLP tasks like search, chatbots, and recommendations.
* The PCA visualization gives a human-readable snapshot of this high-dimensional relationship.

---

### Takeaways

* Word2Vec effectively captures semantic similarities using unsupervised learning.
* PCA helps in understanding word relationships visually.
* Training on a different corpus would yield different semantic patterns, tailored to that domain (e.g., legal, medical, social media).

---

### Requirements

To run the code, make sure the following are installed:

```bash
pip install nltk gensim matplotlib scikit-learn
```

---

### Optional Extensions

* Try using CBOW (`sg=0`) and compare results.
* Visualize more words or cluster them.
* Train on a custom dataset for domain-specific embeddings.
