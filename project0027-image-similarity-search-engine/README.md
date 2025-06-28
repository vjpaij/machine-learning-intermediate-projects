### Description:

An image similarity search engine lets you find images visually similar to a given input. It works by extracting feature vectors from images and comparing them using a similarity metric. In this project, we use pre-trained CNN features from ResNet50 and cosine similarity to match similar images.

- Extracts deep features from images using ResNet50
- Calculates cosine similarity to find visually similar images
- Builds the core of an image search or recommendation engine

## Image Similarity Search using Pre-trained ResNet50

This Python script demonstrates how to perform **image similarity search** using a **pre-trained ResNet50 deep learning model**. It extracts feature vectors from images and compares them using **cosine similarity** to find visually similar images.

### 🧠 Key Concepts

* **Feature Extraction**: Uses ResNet50 (without final classification layer) to convert images into high-dimensional feature vectors.
* **Cosine Similarity**: Measures the cosine of the angle between two feature vectors to determine similarity.
* **Image Retrieval**: Returns top-N similar images to a given query image.

---

### 🧾 Code Walkthrough

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
```

**Imports**: Required libraries for image processing, deep learning (PyTorch), image visualization (Matplotlib), and similarity computation (Scikit-learn).

```python
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
```

* Loads **pre-trained ResNet50** model.
* Removes the final classification layer to get a model that outputs **feature vectors** instead of class labels.
* Sets the model to **evaluation mode**.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Defines a **transform pipeline**:

* Resizes image to 224x224 (expected input for ResNet50).
* Converts to tensor.
* Normalizes using ImageNet statistics (mean and std).

```python
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_t)
    return features.squeeze().numpy()
```

* **extract\_features**: Loads an image, applies the transform, extracts a feature vector using ResNet50, and returns it as a NumPy array.

```python
image_folder = 'images/'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
feature_list = [extract_features(path) for path in image_paths]
```

* Loads all `.jpg` images from the folder.
* Extracts and stores feature vectors of each image.

```python
query_image_path = image_paths[0]
query_feature = extract_features(query_image_path)
similarities = cosine_similarity([query_feature], feature_list)[0]
top_indices = similarities.argsort()[::-1][1:6]
```

* Chooses the first image as the **query image**.
* Computes cosine similarity between query feature and all others.
* Sorts them in descending order and takes top 5 matches (excluding the query itself).

```python
plt.figure(figsize=(12, 3))
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB))
plt.title("Query")
plt.axis('off')

for i, idx in enumerate(top_indices):
    img_path = image_paths[idx]
    img = cv2.imread(img_path)
    plt.subplot(1, 6, i+2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sim {similarities[idx]:.2f}")
    plt.axis('off')

plt.suptitle("Image Similarity Search Results", fontsize=14)
plt.tight_layout()
plt.show()
```

* Displays the **query image** and its **top 5 similar images**.
* Shows similarity scores in the title of each subplot.

---

### 📊 Output Explanation

* The **query image** is displayed on the left.
* The next 5 images are the most similar images based on cosine similarity.
* Each image shows a **"Sim X.XX"** score where 1.00 means highly similar and 0.00 means no similarity.

### ✅ Use Case

This technique can be used in:

* Content-based image retrieval (CBIR)
* Visual search engines
* Duplicate image detection
* Image clustering or recommendation systems

---

### 📌 Note

* The model is trained on ImageNet, so it performs best on natural images.
* For better accuracy, consider fine-tuning the model on your specific dataset.

---

### 📂 Folder Structure

```
project/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── script.py
└── README.md
```

---

### 🛠 Requirements

Install the necessary libraries:

```bash
pip install torch torchvision matplotlib scikit-learn opencv-python pillow
```

---

### 📌 Future Enhancements

* Add GUI to upload and search images.
* Allow batch query comparisons.
* Save feature vectors for faster reuse.
* Use FAISS or Annoy for faster similarity search on large datasets.


