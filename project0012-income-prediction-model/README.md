### Description:

An income prediction model estimates a person’s income based on features such as age, education, occupation, and work hours. This can be used for socioeconomic analysis or targeted marketing. In this project, we build a regression model using scikit-learn to predict annual income based on synthetic data.

- Uses regression modeling to predict income from demographic/work data
- Encodes categorical features and standardizes input
- Evaluates performance with MSE and R², and visualizes results

## Linear Regression on Demographic Data

This script demonstrates a complete pipeline for building and evaluating a linear regression model using a simulated demographic dataset. It includes data preprocessing, feature encoding, feature scaling, model training, prediction, evaluation, and visualization.

---

### Dataset Description

The dataset consists of 10 individuals with the following features:

* **Age**: Numeric (e.g., 25, 32, etc.)
* **Education**: Categorical (High School, Bachelors, Masters, PhD)
* **Occupation**: Categorical (Engineer, Scientist, Professor, Technician)
* **HoursPerWeek**: Numeric (Number of hours worked per week)
* **Income**: Target variable (annual income in USD)

---

### Code Explanation

#### 1. **Data Preparation**

```python
import pandas as pd
import numpy as np
```

* Loads the pandas and numpy libraries for data manipulation.

#### 2. **Preprocessing & Encoding**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

* Label encoding transforms categorical features into integers.
* Standard scaling ensures numeric features are normalized for better model performance.

```python
le_edu = LabelEncoder()
le_occ = LabelEncoder()
df['Education'] = le_edu.fit_transform(df['Education'])
df['Occupation'] = le_occ.fit_transform(df['Occupation'])
```

* 'Education' and 'Occupation' are encoded into numeric values:

  * Education: {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
  * Occupation: {'Engineer': 0, 'Scientist': 1, 'Technician': 2, 'Professor': 3}

#### 3. **Feature Scaling**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* Scales the features to mean=0 and std=1, improving linear regression accuracy.

#### 4. **Train-Test Split**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

* Splits the dataset into 70% training and 30% testing.
* `random_state=42` ensures reproducibility.

#### 5. **Model Training**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

* Trains a linear regression model using the training data.

#### 6. **Prediction and Evaluation**

```python
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* `y_pred`: Model’s predicted income for test data.
* `mse`: Mean Squared Error (lower is better) shows average squared difference between predicted and actual incomes.
* `r2`: R² Score (closer to 1 is better) shows how well the model explains income variability.

Example Output:

```
Predicted Incomes: [91681.97 52560.04 76797.99]
Mean Squared Error: 4581479.56
R² Score: 0.91
```

* The model predicts income quite closely, with R² = 0.91 indicating 91% of the income variation is explained by the model.

#### 7. **Visualization**

```python
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
```

* The scatter plot compares actual vs. predicted incomes.
* The dashed red line represents a perfect prediction (y = x). Points close to this line indicate good predictions.

---

### Summary

This linear regression model demonstrates how to:

* Preprocess a mixed-type dataset
* Train a regression model
* Evaluate its performance
* Visualize prediction quality

A high R² score of 0.91 suggests the model fits the data well. However, the dataset is small and simulated, so real-world application would require more robust validation and more data.

---

> **Note**: This example is educational and may not generalize well due to the small sample size.
