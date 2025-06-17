import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
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
 
print("User-Product Ratings:\n")
print(ratings_df)
 
# Transpose to get product-user matrix
product_user_matrix = ratings_df.T
 
# Compute cosine similarity between products
similarity_matrix = pd.DataFrame(
    cosine_similarity(product_user_matrix),
    index=product_user_matrix.index,
    columns=product_user_matrix.index
)
 
print("\nProduct Similarity Matrix:\n")
print(similarity_matrix.round(2))
 
# ---- Recommendation Function ----
def recommend_products(product_name, similarity_matrix, top_n=3):
    if product_name not in similarity_matrix:
        return "Product not found."
    sorted_similar_products = similarity_matrix[product_name].sort_values(ascending=False)
    recommendations = sorted_similar_products[1:top_n+1]  # Exclude itself
    return recommendations
 
# Example: Recommend similar products to "Product A"
print("\nRecommended products for 'Product A':\n")
print(recommend_products("Product A", similarity_matrix))