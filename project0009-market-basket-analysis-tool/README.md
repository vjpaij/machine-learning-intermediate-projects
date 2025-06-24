### Description:

Market Basket Analysis is a data mining technique used to identify purchase patterns by analyzing customer transactions. This project builds a rule-based tool using the Apriori algorithm to discover frequent itemsets and generate association rules (like "If X is bought, Y is likely to be bought too").

- Converts transactional data into one-hot format
- Applies Apriori algorithm to find frequent itemsets
- Generates association rules with metrics like support, confidence, and lift
- Helps uncover actionable insights for cross-selling

## Apriori Algorithm for Market Basket Analysis

This code demonstrates the use of the Apriori algorithm for **Market Basket Analysis**, a common application of association rule learning. The goal is to discover relationships between items purchased together in transactions.

---

### ✨ Step-by-Step Code Explanation:

#### 🔧 Step 1: Import Libraries

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```

* `pandas` is used for data manipulation.
* `TransactionEncoder` converts transaction data into a format suitable for mining.
* `apriori` finds frequent itemsets based on minimum support.
* `association_rules` derives rules from frequent itemsets using metrics like confidence and lift.

---

#### 👥 Step 2: Define Transactions

```python
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter'],
    ['bread', 'jam'],
    ['milk', 'jam'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'jam'],
    ['butter']
]
```

Each list represents one customer's purchase, i.e., one transaction.

---

#### 🔄 Step 3: One-Hot Encoding

```python
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
```

* Converts item lists into a one-hot encoded DataFrame where:

  * Rows = transactions
  * Columns = items
  * `True` indicates an item was bought in that transaction

Example:

| milk | bread | butter | jam   |
| ---- | ----- | ------ | ----- |
| True | True  | True   | False |

---

#### 🪙 Step 4: Frequent Itemsets via Apriori

```python
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
```

* Finds all itemsets that appear in **≥30%** of transactions.
* `use_colnames=True` keeps item names instead of indices.

---

#### 📈 Step 5: Generate Association Rules

```python
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
```

* Derives rules where the **confidence** is at least 0.6.
* Confidence measures the likelihood of buying `consequents` given `antecedents`.

---

#### 📊 Step 6: Display Rules

```python
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
```

Each rule is explained with:

* **Support**: Fraction of transactions containing both `antecedent` and `consequent`.
* **Confidence**: Likelihood that `consequent` is purchased when `antecedent` is.
* **Lift**: How much more likely `consequent` is given `antecedent` vs random chance.

  * Lift > 1: Positive association
  * Lift = 1: Independent
  * Lift < 1: Negative association

Example Output:

```
If a customer buys [bread], they are likely to buy [butter] (support: 0.50, confidence: 0.71, lift: 1.18)
```

This means:

* 50% of all transactions have both bread and butter.
* 71% of the time when bread is bought, butter is also bought.
* The purchase of butter is 1.18 times more likely when bread is bought, indicating a positive correlation.

---

### 🔍 Business Insight:

* These rules can help in **cross-selling** (e.g., recommend butter when someone buys bread).
* They can optimize **store layout**, **bundling**, and **targeted promotions**.

---

### 🏆 Summary:

This Apriori-based analysis uncovers meaningful patterns in transaction data to guide decision-making in sales and marketing.

---

### 💡 To Run This:

1. Install required package:

```bash
pip install mlxtend
```

2. Run the Python script.
3. Analyze printed association rules for insights.
