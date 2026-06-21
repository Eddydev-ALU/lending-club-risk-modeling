# Predicting Loan Default Risk on the LendingClub Portfolio

A comparison of **classical Machine Learning (Scikit-learn)** and **Deep Learning (TensorFlow)** approaches I used for predicting whether a LendingClub loan will be *charged off* or *fully paid*, using only information available at loan origination.


## Contents

| File | Description |
|------|-------------|
| [`lending_club_default_risk.ipynb`](lending_club_default_risk.ipynb) | Main notebook runs top-to-bottom on Google Colab |

## Problem

Peer-to-peer lending succeeds or fails on credit-risk estimation: *given what is known when a loan is approved, how likely is the borrower to default?* The classes are imbalanced (~20% charge-off) and the cost of errors is asymmetric, so I focused on ROC-AUC, PR-AUC, recall on the default class, and explicit threshold selection not raw accuracy.

## Dataset

**LendingClub accepted loans, 2007–2018 Q4** (2,260,701 loans, 151 raw fields).

- Source: Kaggle `wordsforthewise/lending-club`
  (https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- File used: `accepted_2007_to_2018Q4.csv` (~1.6 GB, **not committed** see below).

I read only the origination-time columns and performed an explicit **data-leakage audit**, excluding post-issuance fields (e.g. `total_pymnt`, `recoveries`, `last_fico_range_high`) that would otherwise reveal the outcome.

## How to run (Google Colab)

1. Open `lending_club_default_risk.ipynb` in Colab (Runtime → GPU recommended but not required).
2. Make the data available the *Data acquisition* cell offers three options (no need to download to your own machine or push the CSV to GitHub):
   - **`kagglehub` (recommended)** one line, `kagglehub.dataset_download('wordsforthewise/lending-club')`, which downloads and caches the data into the Colab runtime (you authenticate with Kaggle once);
   - the **Kaggle CLI** (upload your `kaggle.json` token); or
   - **Google Drive** mount, if you have already copied the file there.
3. `Runtime → Run all`. I set a single random seed, fit all models, build the experiment table, and produce every figure (ROC, PR, confusion matrices, learning curves, calibration).

For a quick run on a free CPU runtime, I train on a reproducible **stratified 250k sample** (`MODEL_SAMPLE`); set `MODEL_SAMPLE = None` to use all ~1.3M resolved loans.

## Methods

- **Classical ML:** Logistic Regression (baseline + balanced/regularised), Random Forest (unconstrained + tuned), Histogram Gradient Boosting.
- **Deep Learning:** TensorFlow **Sequential** API and **Functional** API MLPs with BatchNorm, Dropout, L2, class weighting, early stopping and LR scheduling, fed through a **`tf.data`** pipeline.
- **9 systematically varied experiments** logged into a single comparison table.

## Headline result

On a stratified 250k sample (default rate 19.98%), all nine of my models cluster in a tight **ROC-AUC band of 0.71–0.725**. My best model is the **Functional-API MLP (ROC-AUC 0.725)**, with Histogram Gradient Boosting essentially tied (0.723) the close convergence of very different model families indicates the origination-time *features*, not the algorithm, are the binding constraint. Class weighting is the decisive lever: unweighted models reach ~80% accuracy but catch under 9% of defaults, while weighted models recover 65–71% of defaults at the cost of precision.

## Links

- Dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club/
- GitHub repository: https://github.com/Eddydev-ALU/lending-club-risk-modeling
- Demo video: 
