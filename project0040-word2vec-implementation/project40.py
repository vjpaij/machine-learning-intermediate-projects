import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
 
nltk.download('punkt')
nltk.download('brown')
 
# Load a sample text corpus
sentences = brown.sents()  # Pre-tokenized corpus from NLTK
 
# Train Word2Vec model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, sg=1, workers=4)
 
# Save and load model (optional)
# model.save("word2vec_brown.model")
# model = Word2Vec.load("word2vec_brown.model")
 
# Explore word vectors
print("\nMost similar words to 'money':")
print(model.wv.most_similar('money'))
 
# Visualize a few words in 2D using PCA
words = ['king', 'queen', 'man', 'woman', 'money', 'bank', 'school', 'teacher', 'student', 'war']
word_vecs = [model.wv[word] for word in words]
 
pca = PCA(n_components=2)
result = pca.fit_transform(word_vecs)
 
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.text(result[i, 0] + 0.01, result[i, 1] + 0.01, word)
plt.title("Word2Vec Word Embeddings (PCA Projection)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("word2vec_brown.png", dpi=300, bbox_inches='tight')