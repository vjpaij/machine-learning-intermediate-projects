# Install OpenCV if needed: pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
# Load the image and convert it to RGB
image = cv2.imread('/Users/in22417145/Downloads/White Clean Minimalist LinkedIn Banner.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_shape = image.shape
 
# Reshape image into a 2D array of pixels (rows: pixels, cols: RGB channels)
pixel_data = image.reshape((-1, 3))
 
# Apply K-Means clustering
k = 4  # Number of clusters/segments
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_data)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
 
# Reshape back to original image dimensions
segmented_img = segmented_img.reshape(original_shape).astype(np.uint8)
 
# Display original vs segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title(f"Segmented Image (K={k})")
plt.axis('off')
 
plt.tight_layout()
plt.show()
plt.savefig('segmented_image.png')  # Save the segmented image