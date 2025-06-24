### Description:

Image segmentation is the process of partitioning an image into meaningful segments (like separating objects from background). In this project, we apply K-Means clustering to segment an image based on pixel color similarity — effectively grouping similar colors and simplifying the image.

- Reshapes image data for pixel-wise clustering
- Applies K-Means to segment an image into color clusters
- Visualizes a simplified version of the image based on dominant colors

## K-Means Image Segmentation using OpenCV and scikit-learn

This script demonstrates image segmentation using **K-Means clustering**, a popular unsupervised machine learning technique. It clusters pixels into groups based on color similarity.

### Code Explanation with Reasoning

```python
# Install OpenCV if needed: pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

* **cv2**: Used for image loading and color space conversion.
* **numpy**: Helps reshape and process pixel arrays.
* **matplotlib.pyplot**: Visualizes the results.
* **KMeans from scikit-learn**: Used for clustering pixel colors.

```python
# Load the image and convert it to RGB
image = cv2.imread('sample_image.jpg')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_shape = image.shape
```

* `cv2.imread()` reads the image in BGR format (OpenCV default).
* `cv2.cvtColor(..., COLOR_BGR2RGB)` converts it to RGB (matplotlib compatibility).
* `original_shape` stores dimensions (height, width, color\_channels).

```python
# Reshape image into a 2D array of pixels (rows: pixels, cols: RGB channels)
pixel_data = image.reshape((-1, 3))
```

* Reshapes the image from shape (H, W, 3) to (H\*W, 3), i.e., one row per pixel with 3 color values (R, G, B).
* Required format for clustering where each sample (pixel) is a point in 3D RGB space.

```python
# Apply K-Means clustering
k = 4  # Number of clusters/segments
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_data)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
```

* **k = 4**: You define 4 color clusters (segments).
* `kmeans.fit(...)`: Finds `k` color clusters in the image.
* `kmeans.labels_`: Assigns each pixel to a cluster.
* `kmeans.cluster_centers_`: Returns the RGB value (color) of each cluster center.
* `segmented_img` is a flat array where each pixel is replaced by the color of its assigned cluster center.

```python
# Reshape back to original image dimensions
segmented_img = segmented_img.reshape(original_shape).astype(np.uint8)
```

* Restores the flat segmented pixel data back to original image shape.
* `astype(np.uint8)`: Ensures correct image data type for display.

```python
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
```

* Plots the original and segmented images side by side for visual comparison.

### Result/Output Explanation

* **Original Image**: Shows the actual image as loaded.
* **Segmented Image (K=4)**: Displays the image where each pixel is replaced with its cluster color. This reduces the total number of colors to `k=4`, effectively simplifying the image.
* This can be useful for:

  * Image compression
  * Object segmentation
  * Feature extraction

### Summary

* This project segments an image by grouping similar pixel colors using K-Means clustering.
* Reducing the image to `k` dominant colors can help in computer vision tasks or aesthetic simplification.

### Optional Enhancements

* Experiment with different `k` values.
* Add an elbow method plot to determine optimal `k`.
* Extend to grayscale or use other color spaces like HSV or Lab.
* Save segmented image using `cv2.imwrite()`.

---

**Note**: Always ensure `'sample_image.jpg'` exists in the directory or provide a valid path.
