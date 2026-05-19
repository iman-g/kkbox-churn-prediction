# KKBOX Churn Prediction

![Tests](https://github.com/iman-g/kkbox-churn-prediction/actions/workflows/ci.yml/badge.svg?branch=main)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/dashboard-streamlit-red)](https://kkbox-churn-prediction-ecjvxmyhusiawrcqthqbkd.streamlit.app/)

End-to-end churn prediction pipeline for a music streaming platform, from raw transaction history to a deployed tool that marketing teams can actually use.

**[Live Dashboard →](https://kkbox-churn-prediction-ecjvxmyhusiawrcqthqbkd.streamlit.app/)**  |  **[Medium Article →](https://medium.com/p/ca184844eee5?postPublishedType=initial)**

---

## The Problem

KKBOX needed to identify subscribers likely to cancel before their next renewal. The dataset covers 970K users and 16M subscription transactions. Churn rate is 9%, low enough that a naive model (predict nobody churns) hits 91% accuracy while being completely useless.

---

## Results

| Metric | Value |
|---|---|
| OOF AUC | **0.9171** |
| Average Precision | **0.7174**, 8× above random baseline |
| Precision at operating threshold | 72.9% |
| Recall at operating threshold | 60.0% |
| CV stability (std across 5 folds) | 0.00103 |

At the tuned threshold: for every 100 users flagged for a retention campaign, 73 will actually churn. The remaining 27 false alarms still receive a retention offer, an acceptable trade-off when the cost of missing a churner exceeds the cost of a discounted offer.

---

## Approach

### Phase 1: Survival Analysis

Applied Kaplan-Meier estimation to test whether a data-driven inactivity threshold could replace the industry default of "30 days of inactivity = churned."

Finding: it couldn't, and that's the interesting result. 82.5% of KKBOX subscription periods renew on or before the expiry date, there is no meaningful inactivity gap to model. The survival curve never crosses 50% churn probability within the observable window. This reveals that KKBOX's auto-renewal model makes gap-based survival analysis structurally uninformative, and motivates the binary classification approach in Phase 2. The segmented curves (by subscription frequency) confirm this holds across all user types.

This is the kind of finding that only appears if you actually run the analysis rather than assuming a method will work.

### Phase 2: Binary Classification

Churn is defined as: did this user's subscription lapse in March 2017?

Features are computed from transactions before February 28, 2017, a hard cutoff that prevents any future information from leaking into training. A model built without this cutoff achieved AUC 0.9958, which flagged the leakage. After enforcement: 0.9171.

**Feature groups:**
- **RFM:** recency, frequency, total and average spend
- **Subscription behavior:** cancellation rate, auto-renewal rate, discount rate, plan variety, tenure
- **Member profile:** age, city, registration channel, gender (with explicit missingness flag for the 65% of users with no gender data)

**Model:** XGBoost + LightGBM ensemble, weighted by validation AUC per fold. 5-fold stratified CV preserves the 9% churn rate in every fold.

**Threshold tuning:** the default 0.5 threshold is wrong for a 9% base rate. The operating threshold (0.835) is tuned to achieve 60% recall at maximum precision, reflecting the cost structure of a retention campaign.

---

## Dashboard

Three modes, built for different audiences:

**Campaign Builder** (Marketing)
Score users, filter by risk segment, download a prioritized contact list with recommended intervention per user. Includes a business impact estimator: enter your revenue per user, cost per contact, and expected campaign success rate, get net ROI.

**Cohort Explorer** (Product)
Threshold sensitivity analysis showing how precision, recall, and campaign size change at every threshold value. Score decile breakdown showing actual churn rate per decile.

**Model Explainer** (DS / Leadership)
ROC and precision-recall curves, AUC stability across folds, feature importance, and documented methodology notes explaining the cutoff decision and why stratified CV was used instead of time-series CV.

---

## Running It

**With Docker (recommended):**
```bash
docker build -t kkbox-churn .
docker run -p 8501:8501 kkbox-churn
# Open http://localhost:8501
```

**Without Docker:**
```bash
conda create -n kkbox-churn python=3.10 -y
conda activate kkbox-churn
pip install -r requirements.txt
streamlit run app/dashboard.py
```

**Retrain from scratch** (requires Kaggle data in `data/raw/`):
```bash
python -m scripts.train
```

**Run tests** (no data required):
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## Project Structure

```
├── src/
│   ├── config.py        # paths, hyperparameters, feature cutoff date
│   ├── data.py          # chunk-based loading for large transaction files
│   ├── features.py      # RFM + subscription feature engineering
│   ├── survival.py      # Kaplan-Meier analysis, segmentation, plots
│   ├── model.py         # CV training, threshold tuning, evaluation plots
│   └── evaluation.py    # metrics reporting, JSON persistence
├── app/
│   └── dashboard.py     # Streamlit dashboard (Campaign / Cohort / Explainer)
├── scripts/
│   └── train.py         # end-to-end CLI pipeline
├── tests/
│   └── test_pipeline.py # 13 logic tests, no Kaggle data required
├── results/
│   └── artifacts.pkl    # trained models + OOF predictions (Git LFS)
├── Dockerfile
└── requirements.txt
```

---

## Data

Dataset: [KKBOX Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) (Kaggle)

Raw data is not included in this repository. Download from Kaggle and place CSV files in `data/raw/`. The dashboard runs entirely from `results/artifacts.pkl` and does not require raw data.
