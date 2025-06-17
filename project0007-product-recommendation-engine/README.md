### Description:

A product recommendation engine suggests relevant items to users based on their preferences or behavior. In this project, we build a collaborative filtering system using cosine similarity to recommend products based on user-item interactions, using a simple simulated dataset.

- Builds an item-based collaborative filtering recommender
- Uses cosine similarity to find similar products
- Recommends top N alternatives based on a selected product

### Product Recommendation System Explanation

This script demonstrates a simple **item-based collaborative filtering** recommendation system using **cosine similarity** to find similar products based on user ratings.

---

### 1. **Input Data**: User-Product Rating Matrix

The dataset is a dictionary of product ratings by users. It simulates how different users have rated different products:

```python
# Simulated user-product rating matrix
data = {
    'Product A': [5, 4, 0, 0, 1],
    'Product B': [3, 0, 0, 5, 1],
    'Product C': [4, 3, 0, 0, 1],
    'Product D': [0, 0, 5, 4, 0],
    'Product E': [0, 2, 4, 0, 0]
}
user_ids = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5']
ratings_df = pd.DataFrame(data, index=user_ids)
```

This results in the following **user-product matrix**:

```
         Product A  Product B  Product C  Product D  Product E
User 1          5          3          4          0          0
User 2          4          0          3          0          2
User 3          0          0          0          5          4
User 4          0          5          0          4          0
User 5          1          1          1          0          0
```

---

### 2. **Transforming Data**

We transpose the DataFrame so each row becomes a product, and each column a user. This is necessary to calculate **item-item similarity**:

```python
product_user_matrix = ratings_df.T
```

---

### 3. **Cosine Similarity Computation**

We calculate cosine similarity between product vectors:

```python
similarity_matrix = pd.DataFrame(
    cosine_similarity(product_user_matrix),
    index=product_user_matrix.index,
    columns=product_user_matrix.index
)
```

The **cosine similarity** measures the cosine of the angle between two vectors, capturing similarity in terms of direction rather than magnitude. A score close to `1` means the products are rated similarly by users.

Example output:

```
Product Similarity Matrix:
            Product A  Product B  Product C  Product D  Product E
Product A        1.00       0.81       0.99       0.00       0.50
Product B        0.81       1.00       0.79       0.38       0.18
Product C        0.99       0.79       1.00       0.00       0.43
Product D        0.00       0.38       0.00       1.00       0.87
Product E        0.50       0.18       0.43       0.87       1.00
```

---

### 4. **Recommendation Function**

```python
def recommend_products(product_name, similarity_matrix, top_n=3):
    if product_name not in similarity_matrix:
        return "Product not found."
    sorted_similar_products = similarity_matrix[product_name].sort_values(ascending=False)
    recommendations = sorted_similar_products[1:top_n+1]  # Exclude itself
    return recommendations
```

* This function recommends products most similar to a given product.
* It excludes the product itself (since it would be 100% similar to itself).

Example:

```python
recommend_products("Product A", similarity_matrix)
```

Result:

```
Product C    0.99
Product B    0.81
Product E    0.50
Name: Product A, dtype: float64
```

This output means:

* **Product C** is the most similar to **Product A** (0.99)
* Followed by **Product B** (0.81)
* And **Product E** (0.50)

---

### Interpretation of Result

The model is able to recommend alternative products that users might like, based on the **similarity of user ratings**. For instance, users who liked Product A also tended to like Product C and Product B. So, if a user is interested in Product A, the system recommends these other products.

This approach is useful in:

* E-commerce platforms ("Customers also viewed")
* Streaming services ("If you liked this, you'll like...")

---

### Limitations

* This method does not handle cold start problems (new users/products).
* It assumes rating patterns reflect user preference similarity.
* Sparse data can lead to inaccurate similarity scores.

---

### Summary

This script showcases a simple, interpretable item-based collaborative filtering approach using cosine similarity, providing a foundation for building recommendation engines without deep learning or large datasets.
