### Description:

An A/B testing analyzer helps evaluate whether two variants (A and B) have significantly different performance, such as conversion rates. In this project, we build a tool that compares outcomes using a two-proportion z-test to determine if the difference between versions is statistically significant.

- Compares two conversion rates using z-test for proportions
- Calculates and interprets p-values and statistical significance
- Visualizes results for quick decision-making

## A/B Test Analysis Using Two-Proportion Z-Test

### Overview

This code performs a **two-proportion Z-test** to compare the conversion rates between two variants in an A/B test — Variant A (control group) and Variant B (test group). The objective is to statistically determine whether the difference in conversion rates is significant.

---

### Code Walkthrough & Explanation

#### 1. **Importing Libraries**

```python
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
```

* `proportions_ztest`: From the `statsmodels` package, used to conduct a Z-test for two proportions.
* `matplotlib.pyplot`: For visualizing the conversion rate comparison between the two variants.

---

#### 2. **Sample Data Setup**

```python
conversions_A = 120
visitors_A = 2400
conversions_B = 150
visitors_B = 2300
```

* **Variant A (Control Group)**:

  * 120 users converted out of 2400 total.
* **Variant B (Test Group)**:

  * 150 users converted out of 2300 total.

This data simulates an experiment to compare conversion performance.

---

#### 3. **Preparing Data for Z-Test**

```python
counts = [conversions_A, conversions_B]  # successes
nobs = [visitors_A, visitors_B]          # total observations
```

These lists prepare inputs for the statistical test:

* `counts`: Number of conversions (successes) for both groups.
* `nobs`: Total visitors (observations) for both groups.

---

#### 4. **Running the Z-Test**

```python
z_stat, p_value = proportions_ztest(count=counts, nobs=nobs)
```

* `z_stat`: The Z statistic measures how far the observed difference in proportions is from the null hypothesis (no difference), in units of standard error.
* `p_value`: The probability of observing a result as extreme as this one (or more) if the null hypothesis were true.

---

#### 5. **Displaying Results**

```python
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

These print statements output the numerical test results.

```python
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference between A and B (Reject Null Hypothesis)")
else:
    print("Result: No statistically significant difference (Fail to Reject Null Hypothesis)")
```

* `alpha = 0.05`: The significance level (5%). If `p_value < alpha`, the result is considered statistically significant.
* If significant: Conclude that Variant B performs differently from Variant A.

---

#### 6. **Visualizing Conversion Rates**

```python
labels = ['Variant A', 'Variant B']
conversion_rates = [conversions_A / visitors_A, conversions_B / visitors_B]
```

Calculates and labels conversion rates:

* Variant A: 120 / 2400 = 0.05 (5%)
* Variant B: 150 / 2300 ≈ 0.0652 (6.52%)

```python
plt.bar(labels, conversion_rates, color=['blue', 'green'])
plt.title('Conversion Rate Comparison: A/B Test')
plt.ylabel('Conversion Rate')
plt.ylim(0, max(conversion_rates) + 0.02)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

This creates a bar chart showing the conversion rates, making the difference visually clear.

---

### Interpretation of Output

Example Output (will vary with data):

```
Z-statistic: 2.2361
P-value: 0.0253
Result: Statistically significant difference between A and B (Reject Null Hypothesis)
```

* **Z-statistic = 2.2361**: The observed difference in conversion rates is over 2 standard errors away from the null.
* **P-value = 0.0253**: Less than the threshold of 0.05.
* **Conclusion**: Variant B shows a statistically significant improvement in conversion rate compared to Variant A.

---

### Summary

* This script uses a **two-proportion Z-test** to compare conversion rates.
* A **low p-value (< 0.05)** indicates that the improvement in conversions in Variant B is unlikely due to chance.
* Visualization helps reinforce the statistical finding.

This analysis is commonly used in A/B testing scenarios such as website optimization, feature rollout, and product experiments.

