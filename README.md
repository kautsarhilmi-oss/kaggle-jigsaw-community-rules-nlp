# 🛡️ Community Rules Violation Detection — Kaggle Jigsaw NLP

[![Kaggle](https://img.shields.io/badge/Kaggle-Jigsaw%20Agile%20Community%20Rules-blue)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![CV AUC](https://img.shields.io/badge/CV%20AUC-0.7929-brightgreen)]()
[![Task](https://img.shields.io/badge/Task-NLP%20Classification-purple)]()

## Overview

Binary NLP classification to predict whether a Reddit comment violates
a community-specific rule — helping moderators scale content moderation
across diverse subreddits with different norms and expectations.

- **Competition:** Jigsaw — Agile Community Rules Classification
- **Task:** Predict probability of `rule_violation` (0 or 1)
- **Metric:** ROC-AUC
- **CV Score:** 0.7929 (5-Fold Stratified CV)
- **Public LB:** 0.62919 | **Private:** 0.64757
- **Model:** TF-IDF + Logistic Regression

---

## What Makes This Problem Unique

Unlike standard text classification, this task requires **zero-shot
generalization** — the test set contains rules that never appear
in training. The model must understand whether a comment violates
a rule it has never seen before, using only the rule description
and few-shot examples as context.

---

## Dataset

| Split | Rows | Features | Notes |
|-------|------|----------|-------|
| Train | 2,029 | 8 | Only 2 rules in training |
| Test  | 10   | 8 | Contains unseen rules |

**Columns:**
- `body` — the Reddit comment text
- `rule` — the community rule being evaluated
- `subreddit` — which subreddit (100 unique communities)
- `positive_example_1/2` — example comments that DO violate the rule
- `negative_example_1/2` — example comments that DON'T violate the rule

---

## Approach

### 1. Few-Shot Context Engineering
The key insight: `positive_example` and `negative_example` columns
act as **few-shot demonstrations**. By computing word overlap between
the body and these examples, we extract a behavioral signal about
whether the comment resembles known violations.

### 2. Text Feature Construction
Combined multiple text fields into structured representations:
- `body_rule`: comment + rule definition
- `full_context`: all 6 text fields concatenated with [SEP] tokens

### 3. Overlap Features (Domain Signal)
```python
overlap_pos = jaccard_similarity(body, positive_example)
overlap_neg = jaccard_similarity(body, negative_example)
violation_signal = overlap_pos - overlap_neg
```
A positive `violation_signal` means the comment resembles
known violations more than known non-violations.

### 4. TF-IDF Representations
- `body_rule`: 5,000 features, 1-2 grams
- `full_context`: 3,000 features, 1-2 grams
- Combined with numeric overlap features via sparse matrix stacking

### 5. Logistic Regression Baseline
Strong interpretable baseline for NLP — with sublinear TF-IDF
and L2 regularization. 5-Fold Stratified CV for robust evaluation.

---

## Results

| Fold | AUC |
|------|-----|
| 1 | 0.82752 |
| 2 | 0.76146 |
| 3 | 0.79546 |
| 4 | 0.80383 |
| 5 | 0.77641 |
| **Mean CV** | **0.79293 ± 0.02274** |
| **Public LB** | **0.62919** |
| **Private** | **0.64757** |

**Note on generalization gap:** The CV-to-LB gap reflects the
core challenge of this competition — the test set contains
rules unseen during training. The model must generalize to
new community norms using only the rule description and
few-shot examples, which is inherently harder than standard
i.i.d. classification.

---

## Key Learnings

- **Few-shot context as features** — positive/negative examples
  provide signal even as simple word overlap metrics
- **Zero-shot generalization is hard** — a model trained on 2 rules
  must generalize to unseen rules purely from description
- **Domain shift matters** — CV score and LB score diverge
  when test distribution differs from training
- **NLP + tabular hybrid** — combining TF-IDF with hand-crafted
  overlap features outperforms pure TF-IDF

---

## Potential Improvements

- Sentence Transformers (semantic similarity vs word overlap)
- Cross-encoder models for body-rule pair scoring
- LLM-based zero-shot classification (GPT, Llama)
- Prompt engineering using the few-shot examples directly

---

## How to Run

```bash
pip install scikit-learn pandas numpy scipy

# On Kaggle Notebook (data auto-available):
# /kaggle/input/jigsaw-agile-community-rules/

# Locally:
# Download from https://www.kaggle.com/competitions/jigsaw-agile-community-rules/data
python jigsaw_community_rules.py
```

---

## Author

**Kautsar Hilmi** — Magister Teknik Informatika, Universitas Pamulang
*Data Science | NLP | Machine Learning*

[![GitHub](https://img.shields.io/badge/GitHub-kautsarhilmi--oss-black)](https://github.com/kautsarhilmi-oss)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kautsarhilmi-blue)](https://www.linkedin.com/in/kautsarhilmi/)
