### Description:

A customer segmentation tool groups customers based on shared characteristics such as behavior, demographics, or purchase history. This enables targeted marketing and personalized experiences. In this project, we use K-Means clustering on synthetic customer data to identify distinct customer segments.

- Groups customers into segments using K-Means clustering
- Preprocesses data with standard scaling
- Visualizes high-dimensional clusters with PCA

## Customer Segmentation Using KMeans Clustering

This script demonstrates how to apply machine learning techniques to segment customers based on their demographics and behavior. The process includes data generation, preprocessing, clustering using KMeans, dimensionality reduction for visualization, and plotting the results.

### 🔍 Step-by-Step Explanation

#### 1. **Importing Required Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

These libraries serve the following purposes:

* `pandas`, `numpy`: Data manipulation and numerical operations.
* `matplotlib.pyplot`: Visualization.
* `KMeans`: Clustering algorithm from `sklearn`.
* `StandardScaler`: Standardizes features.
* `PCA`: Reduces dimensions for better visualization.

#### 2. **Simulated Customer Data**

```python
np.random.seed(42)
data = {
    'Age': np.random.randint(20, 65, 100),
    'AnnualIncome': np.random.randint(20000, 120000, 100),
    'SpendingScore': np.random.randint(1, 100, 100)
}
df = pd.DataFrame(data)
```

This block creates a DataFrame of 100 fake customers with three features:

* `Age`: Random ages between 20 and 65.
* `AnnualIncome`: Random incomes between \$20,000 and \$120,000.
* `SpendingScore`: A synthetic score (e.g., loyalty or spending frequency) between 1 and 100.

#### 3. **Standardization**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

Standardization ensures each feature contributes equally to the clustering, by scaling them to have mean = 0 and std = 1.

#### 4. **KMeans Clustering**

```python
k = 4
model = KMeans(n_clusters=k, random_state=42)
df['Segment'] = model.fit_predict(X_scaled)
```

* We set `k=4`, indicating the desired number of customer segments.
* `fit_predict` assigns a cluster (segment) label to each customer.
* These segment labels are added to the original DataFrame.

#### 5. **PCA for Visualization**

```python
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)
```

* Reduces high-dimensional standardized data to 2 dimensions.
* Makes it easier to visualize customer segments on a 2D plot.

#### 6. **Plotting Segments**

```python
plt.figure(figsize=(8, 5))
for cluster in range(k):
    plt.scatter(pca_features[df['Segment'] == cluster, 0], pca_features[df['Segment'] == cluster, 1], label=f'Segment {cluster}')
plt.title('Customer Segmentation (K-Means + PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

* A scatter plot is generated for each cluster.
* Customers belonging to the same cluster are shown in the same color.
* This helps to visually inspect how well the clustering has worked.

### 📈 Results and Interpretation

* **df\['Segment']**: A new column indicating which cluster/segment each customer belongs to (from 0 to 3).
* **Visualization**: The PCA plot shows 4 distinct customer segments in reduced 2D space.

### ✅ What Does It Mean?

* This approach helps businesses identify patterns in customer behavior.
* E.g., one cluster may contain young high-spenders, another may include older low-spenders, etc.
* These insights can guide marketing strategies, personalized offers, and business decisions.

### 💡 Notes

* In real scenarios, we use actual customer data.
* Choosing the right value of `k` (number of clusters) can be guided by the Elbow Method or Silhouette Score.
* PCA is only used for visualization, not clustering.

---

This notebook is a simple yet powerful example of unsupervised learning using KMeans for customer segmentation.
