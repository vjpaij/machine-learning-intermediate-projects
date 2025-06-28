import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
 
# Load sample dataset (Iris)
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
 
# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
 
# Initialize Self-Organizing Map
som_size = 10  # 10x10 grid
som = MiniSom(x=som_size, y=som_size, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=100)
 
# Plot SOM with labels
plt.figure(figsize=(8, 8))
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Get winning node
    plt.text(w[0], w[1], str(y[i]), color=plt.cm.Set1(y[i] / 3.0), fontdict={'weight': 'bold', 'size': 12})
plt.title("Self-Organizing Map (Iris Data)")
plt.xlim([0, som_size])
plt.ylim([0, som_size])
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig("som_iris.png", dpi=300)