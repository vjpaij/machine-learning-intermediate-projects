import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load pre-trained ResNet50 model (without final classification layer)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
resnet.eval()  # Set to evaluation mode

# Transformation pipeline (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Helper function: preprocess and extract features
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(img_t)  # Extract features
    return features.squeeze().numpy()  # Convert to NumPy vector

# Load images from directory
image_folder = 'images/'  # Replace with your folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

# Extract features from all images
feature_list = [extract_features(path) for path in image_paths]

# Choose a query image
query_image_path = image_paths[0]
query_feature = extract_features(query_image_path)

# Compute cosine similarity
similarities = cosine_similarity([query_feature], feature_list)[0]
top_indices = similarities.argsort()[::-1][1:6]  # Top 5 matches excluding query

# Display results
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
plt.savefig('similarity_search_results.png', dpi=300, bbox_inches='tight')