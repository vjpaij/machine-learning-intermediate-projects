### Description:

A resume screening system automates the evaluation of resumes to identify the most relevant candidates for a job. It compares resumes to job descriptions using text vectorization and cosine similarity. In this project, we simulate this process using TF-IDF and rank resumes by relevance to a job post.

- Compares resumes to job descriptions using TF-IDF and cosine similarity
- Ranks candidates based on semantic relevance
- Builds a foundation for automated resume screening tools

## Resume Matcher using TF-IDF and Cosine Similarity

### 📌 Overview

This script demonstrates a simple, effective method to match job descriptions with candidate resumes using natural language processing (NLP). The primary techniques used are:

* **TF-IDF Vectorization**: Converts text into numerical vectors.
* **Cosine Similarity**: Measures how similar two text vectors are, regardless of their magnitude.

This tool can help automate resume screening by ranking candidates based on textual relevance to a job description.

---

### 🧠 Code Explanation

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

* **pandas**: (Not used in this code, but commonly used for data handling)
* **TfidfVectorizer**: Converts text into TF-IDF vectors, reducing the impact of common words.
* **cosine\_similarity**: Calculates similarity between vectors from 0 (not similar) to 1 (identical).

```python
# Job description text
job_description = """
Looking for a Python developer with experience in machine learning, data analysis, and web development.
Familiarity with Django, pandas, and scikit-learn is preferred. Strong problem-solving and communication skills are essential.
"""
```

* A **sample job description** is defined. In real-world applications, this could come from a database or user input.

```python
# Sample candidate resumes
resumes = [
    "Experienced Python developer with strong background in Django and web APIs. Built scalable applications.",
    "Data scientist skilled in Python, pandas, and machine learning. Worked on several classification problems.",
    "Frontend developer with React and JavaScript expertise. Limited experience with backend systems.",
    "Software engineer familiar with Java and Spring Boot. Looking to switch to data science roles.",
    "Machine learning enthusiast with academic projects in scikit-learn and deep learning using PyTorch."
]
```

* Each resume is simulated as a string. In production, you'd parse these from PDF or Word documents.

```python
# Combine documents for vectorization
documents = [job_description] + resumes
```

* The job description and all resumes are combined into a single list to be processed together.

```python
# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
```

* Text is transformed into a **sparse matrix** of TF-IDF features.
* `stop_words='english'` removes common words like "the", "is", "in", etc.

```python
# Extract individual vectors
job_vector = tfidf_matrix[0]             # Job description vector
resume_vectors = tfidf_matrix[1:]        # Resumes vectors
```

* The first row corresponds to the job description. Remaining rows represent resumes.

```python
# Compute cosine similarity
similarities = cosine_similarity(job_vector, resume_vectors).flatten()
```

* Computes similarity between job description and each resume.
* Result: A 1D array of similarity scores.

```python
# Rank resumes by similarity score
ranked_resumes = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
```

* Each resume is assigned an index and its similarity score.
* Sorted in descending order of similarity.

```python
# Output top matches
for idx, score in ranked_resumes:
    print(f"Resume {idx + 1} - Similarity Score: {score:.2f}")
    print(resumes[idx])
    print("-" * 60)
```

* Prints ranked resumes with their similarity scores for easy review.

---

### 📈 Result Interpretation

Each resume gets a **similarity score (0 to 1)** that reflects how closely its content matches the job description:

* **Higher score (e.g., 0.85)**: Very relevant.
* **Mid-range score (e.g., 0.5)**: Some relevance, but gaps exist.
* **Lower score (e.g., <0.3)**: Likely unrelated.

This approach gives a **quantitative, reproducible way** to shortlist resumes based on keyword and contextual relevance.

---

### ✅ Use Case Applications

* Resume filtering tools
* Candidate-job matching engines
* HR automation
* Text similarity engines

---

### 🚀 Improvements

* Parse real resumes from PDFs using libraries like `pdfminer` or `PyMuPDF`
* Include named entity recognition (NER)
* Weight domain-specific terms
* Use semantic similarity models like BERT for better results
