import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Generate synthetic dataset with 4 classes
X, y = make_classification(n_samples=500, n_features=10, n_informative=6, 
                           n_classes=4, n_clusters_per_class=1, random_state=42)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initialize and train a multi-class logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
 
# Predict on test set
y_pred = model.predict(X_test)
 
# Evaluate performance
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Set2')
plt.title("Confusion Matrix - Multi-Class Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png")