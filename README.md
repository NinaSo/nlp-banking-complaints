# NLP Pipeline for Banking Complaint Analysis

End-to-end NLP pipeline applied to ~7,000 real banking complaints (2023). Covers automated department routing via text classification, lexicon-based sentiment scoring with VADER, and transformer-based sentiment inference with DistilBERT.

## Project Overview

| Step | Description |
|------|-------------|
| Text preprocessing | Lowercasing, number/punctuation removal, tokenization, stopword removal, lemmatization (NLTK) |
| Feature engineering | TF-IDF vectorization (top 5,000 features) |
| Department classification | Multinomial Naive Bayes and Logistic Regression (multi-class) |
| Sentiment — VADER | Rule-based compound scoring; department-level aggregation |
| Sentiment — BERT | DistilBERT fine-tuned on SST-2; confidence-thresholded 3-class mapping |

## Dataset

`complaints_banking_2023.csv` — 7,011 banking complaints filed January–October 2023, including:
- Complaint description (free text)
- Department (CASA, Credit Cards, Mortgage, Loans, Credit Reports, Remittance, Others)
- Bank response, state, ZIP code, date received

> Note: bank names and personal identifiers in complaint text have been anonymized (replaced with `XXXX`).

## Key Results

### Department Classification

| Model | Accuracy |
|-------|----------|
| Multinomial Naive Bayes | ~68.5% |
| Logistic Regression | ~75.0% |

Logistic Regression showed stronger performance across major departments (Mortgage, Loans, Credit Reports). Both models struggled on minority classes (Remittance, Others) — a known effect of class imbalance in multi-class text classification.

### Sentiment Analysis

- **VADER**: Compound scores revealed widespread negative sentiment, with variation across departments — enabling severity-based complaint prioritization.
- **DistilBERT**: Classified the vast majority of complaints as NEGATIVE with high confidence (>0.99), consistent with the grievance nature of the dataset. Disagreements with VADER (VADER-positive / BERT-negative) corresponded to factually-resolved but grievance-framed complaints.

### Combined Insight

VADER and BERT are complementary:
- VADER provides a **graded severity score** — useful for escalation routing.
- BERT provides **contextual framing detection** — useful for bulk classification.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook banking_complaints_nlp.ipynb
```

### Requirements

```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
transformers
torch
vaderSentiment
```

## File Structure

```
repo_nlp_banking_complaints/
├── banking_complaints_nlp.ipynb   # Full notebook: EDA → preprocessing → classification → VADER → BERT
├── complaints_banking_2023.csv    # Banking complaints dataset (Jan–Oct 2023)
└── README.md
```

## Business Applications

| Capability | Application |
|-----------|-------------|
| Department classification | Auto-triage incoming complaints to the right team |
| VADER compound score | Severity-based escalation (very negative → critical queue) |
| BERT sentiment | Robust grievance detection at scale |
| Department-level VADER aggregation | Identify product areas with highest customer dissatisfaction |
