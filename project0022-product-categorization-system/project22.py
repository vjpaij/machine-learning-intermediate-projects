import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated product data
data = {
    'ProductName': [
        'iPhone 13 Pro Max 256GB',
        'Leather Office Chair Ergonomic',
        'Samsung Galaxy Watch 5',
        'Nike Running Shoes',
        'USB-C Charger Adapter 20W',
        'Wooden Dining Table 6 Seater',
        'Bluetooth Wireless Earbuds',
        'HP Pavilion Gaming Laptop',
        'Electric Kettle Stainless Steel',
        'Canon DSLR Camera Lens Kit'
    ],
    'Category': [
        'Electronics',
        'Furniture',
        'Electronics',
        'Footwear',
        'Electronics',
        'Furniture',
        'Electronics',
        'Electronics',
        'Appliances',
        'Electronics'
    ]
}
 
df = pd.DataFrame(data)
 
# Convert categories to numeric labels
df['Category_Label'] = df['Category'].astype('category').cat.codes
label_map = dict(enumerate(df['Category'].astype('category').cat.categories))
 
# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ProductName'])
y = df['Category_Label']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
label = list(label_map.keys())
print("Classification Report:\n")
print(classification_report(y_test, y_pred, labels=label, target_names=label_map.values()))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=label)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Confusion Matrix - Product Categorization")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.show()
plt.savefig('product_categorization_confusion_matrix.png')  # Save the confusion matrix plot