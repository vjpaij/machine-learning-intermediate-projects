### Description:

In this project, we build a basic image classifier using scikit-learn. We'll use the Digits dataset, which contains 8x8 pixel images of handwritten digits (0–9). We extract pixel features and train a Support Vector Machine (SVM) classifier to recognize the digits.

- Converts image data into features (pixel values)
- Trains an SVM classifier for multi-class classification
- Evaluates results with classification report and confusion matrix
- Visualizes both input data and performance metrics

## Digit Classification using SVM - Project Overview

This script demonstrates how to use a Support Vector Machine (SVM) for classifying handwritten digits from the popular `digits` dataset provided by `scikit-learn`. The dataset consists of 8x8 grayscale images of digits (0-9), and the model aims to accurately predict the digit labels.

### Libraries Used

* `sklearn.datasets`: For loading the digits dataset.
* `sklearn.model_selection.train_test_split`: For splitting data into training and test sets.
* `sklearn.preprocessing.StandardScaler`: For feature scaling.
* `sklearn.svm.SVC`: For building the SVM model.
* `sklearn.metrics`: For evaluating the model's performance.
* `matplotlib.pyplot`: For visualizing the data and results.

### Step-by-Step Explanation

#### 1. **Load the Dataset**

```python
from sklearn import datasets
```

The `digits` dataset is loaded using `datasets.load_digits()`, which includes 1,797 samples of 8x8 pixel images.

```python
digits = datasets.load_digits()
X = digits.data          # Flattened image pixels (64 features)
y = digits.target        # Corresponding digit labels (0–9)
```

#### 2. **Visualize Sample Images**

```python
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("Sample Images from Digits Dataset")
plt.tight_layout()
plt.show()
```

Five sample images are displayed to give a visual sense of the dataset.

#### 3. **Split Dataset**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Splits the data into 70% training and 30% testing sets.

#### 4. **Scale Features**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardizes the features to have zero mean and unit variance, which is important for SVM performance.

#### 5. **Train the SVM Model**

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', gamma=0.001, C=10)
model.fit(X_train_scaled, y_train)
```

An SVM classifier with an RBF kernel is trained on the scaled training data.

* `C=10`: Regularization parameter.
* `gamma=0.001`: Kernel coefficient.

#### 6. **Make Predictions and Evaluate**

```python
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
```

Displays precision, recall, and F1-score for each class.

#### 7. **Visualize Confusion Matrix**

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.show()
```

The confusion matrix is plotted to visualize how well the model performs for each digit.

### Conclusion

This script offers a complete pipeline for training an SVM to classify handwritten digits, including preprocessing, model training, evaluation, and visualization. It demonstrates the effectiveness of SVMs on simple image recognition tasks.
