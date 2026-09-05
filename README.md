# Churn & Reactivation Decision Engine

An end-to-end banking analytics and machine-learning project that identifies customers at risk of **behavioural disengagement**, ranks them by risk, and converts model output into practical reactivation targeting decisions.

The project uses the real anonymized **Berka / PKDD'99 Czech financial dataset**, with more than one million banking transactions across 4,500 accounts.

## Project objective

The source data does not contain a direct customer-churn label. Rather than inventing account closures, this project defines a forward-looking **behavioural disengagement** target from observed transaction activity.

The workflow answers four questions:

1. Which active customers are most likely to show a severe decline in activity over the next 90 days?
2. Which model ranks those customers most effectively?
3. How many true disengagement cases can be captured by targeting only a small share of customers?
4. Under explicit campaign assumptions, when could a reactivation intervention break even?

## Dataset

**Berka / PKDD'99 Financial Dataset**

- 4,500 accounts
- 5,369 clients
- 1,056,320 transactions
- 892 cards
- 682 loans
- Transaction history: 1993-01-01 to 1998-12-31

Data source used for this project:

- Kaggle mirror: https://www.kaggle.com/datasets/marceloventura/the-berka-dataset
- CTU Relational Learning Repository: https://relational.fel.cvut.cz/dataset/Financial

Raw source files are intentionally excluded from Git. The repository contains the code required to build the modelling dataset locally.

## Target definition

A literal "account disappears" churn label was unsuitable because almost every account remained active through the end of the source period. Simple inactivity and fixed percentage-drop labels were also tested and rejected because they were either too rare or strongly distorted by monthly seasonality.

The final target is **seasonally adjusted behavioural disengagement**.

For each monthly snapshot from January 1997 through September 1998:

1. Keep accounts with at least **4 customer-initiated transactions in the previous 90 days**.
2. Count customer-initiated transactions in the previous 90 days and the following 90 days.
3. Calculate each account's future-to-past activity ratio.
4. Calculate the median future-to-past ratio across eligible accounts at that same snapshot.
5. Label an account as disengaged when:

```text
account future/past activity ratio
    <= 0.60 × same-snapshot median future/past activity ratio
```

Customer-initiated operations are:

- `VYBER`
- `PREVOD NA UCET`
- `VYBER KARTOU`
- `VKLAD`

This produces a difficult but workable imbalanced classification problem without treating normal seasonal slowdowns as churn.

## Modelling dataset

The feature builder creates **82,966 account-month observations** with 100 predictive features derived only from information available at each snapshot.

Examples include transaction recency, 30-day and 90-day activity, incoming and outgoing flows, balances, cash withdrawals and deposits, transfer behaviour, operation diversity, card and loan engagement, account age, and recent activity changes.

## Time-based validation

| Split | Period | Rows | Positives | Target rate |
|---|---|---:|---:|---:|
| Train | 1997 snapshots | 43,959 | 1,131 | 2.57% |
| Validation | Jan-Jun 1998 | 25,830 | 737 | 2.85% |
| Test | Jul-Sep 1998 | 13,177 | 199 | 1.51% |

The test period is treated as a locked holdout and is not used for model or targeting-policy tuning.

## Models compared

| Model | ROC-AUC | PR-AUC | Top-10% precision | Top-10% lift |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8631 | 0.1431 | 14.32% | 5.02x |
| Random Forest | 0.8997 | 0.1805 | 15.95% | 5.59x |
| **Gradient Boosting** | **0.9106** | **0.2159** | **17.69%** | **6.20x** |

Because the positive class is rare, **PR-AUC, precision, recall and lift** are emphasized over accuracy.

## Final holdout performance

| Metric | Final test result |
|---|---:|
| ROC-AUC | **0.9227** |
| PR-AUC | **0.1983** |
| Test prevalence | 1.51% |
| Top-10% precision | **10.70%** |
| Top-10% recall | **70.85%** |
| Top-10% lift | **7.08x** |
| Cases captured | **141 / 199** |

Targeting only the highest-risk **10% of customers captured 70.85% of all observed disengagement cases** in the out-of-time test period.

## Targeting trade-off

| Targeted share | Precision | Recall | Lift |
|---|---:|---:|---:|
| 1% | 32.43% | 11.40% | 11.37x |
| 5% | 23.53% | 41.25% | 8.25x |
| **10%** | **17.69%** | **62.01%** | **6.20x** |
| 15% | 14.58% | 76.66% | 5.11x |
| 20% | 12.12% | 84.94% | 4.25x |
| 25% | 10.28% | 90.09% | 3.60x |
| 30% | 8.96% | 94.17% | 3.14x |

The 10% tier was selected on validation as a practical balance between campaign reach and case capture.

## Intervention economics

The source dataset contains no historical retention campaigns, treatment effects or customer lifetime value. Therefore, campaign economics are **illustrative scenario analysis**, not observed business results.

Example assumptions:

- cost per contact: **2.00 value units**
- save rate among truly at-risk customers: **20%**

Under those assumptions, the validation top-10% tier requires approximately **56.52 value units of retained-customer value per successful save to break even**.

## Feature importance

Permutation importance on the validation period identified these leading predictive signals:

1. `cash_withdrawals_90d`
2. `outgoing_transfers_30d`
3. `recency_days_prevprev30d`
4. `unique_operations_90d`
5. `customer_amount_90d`
6. `incoming_amount_30d`
7. `max_amount_90d`
8. `all_tx_count_90d`

These are predictive associations, not causal effects.

## High-risk customer profile

Compared with the remaining 90% of validation observations, the top-10% risk group showed:

- median cash withdrawals over 90 days: **12 vs 8**
- median unique operation types: **2 vs 3**
- median 30-day balance change: **-2,543.90 vs +1,095.00**
- fewer outgoing transfers
- fewer cash deposits

The top-10% group had an observed disengagement rate of **17.69%**, compared with **1.20%** among the remaining 90%.

## Reactivation action framework

| Action hypothesis | Validation customers | Observed disengagement rate |
|---|---:|---:|
| Account health check-in | 983 | 15.16% |
| Payments re-engagement | 616 | 19.16% |
| General reactivation | 436 | 17.66% |
| Product-usage re-engagement | 293 | 18.09% |
| Deposit re-engagement | 255 | **23.53%** |

These labels are campaign-design hypotheses. The dataset contains no randomized intervention data, so the project does not claim that any action causes retention.

## SQL workflow

`src/load_sqlite.py` loads the eight Berka source tables into SQLite and creates indexes for transaction analysis.

`sql/01_source_profile.sql` demonstrates SQL-based source validation, transaction-period checks, operation distributions, and customer-initiated activity aggregation.

Verified source counts:

```text
account       4,500
card            892
client        5,369
disp          5,369
district         77
loan            682
order_table   6,471
trans     1,056,320
```

## Repository structure

```text
churn-reactivation-engine/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── sql/
│   └── 01_source_profile.sql
├── src/
│   ├── build_features.py
│   ├── load_sqlite.py
│   ├── train_baseline.py
│   ├── train_models.py
│   ├── evaluate_thresholds.py
│   ├── evaluate_economics.py
│   ├── evaluate_test.py
│   ├── analyze_feature_importance.py
│   ├── profile_risk_segments.py
│   └── build_reactivation_actions.py
├── dashboard/
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

## Reproduce the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Berka CSV files in `data/raw/berka/`, then run:

```bash
python src/build_features.py
python src/train_models.py
python src/evaluate_thresholds.py
python src/evaluate_economics.py
python src/evaluate_test.py
python src/analyze_feature_importance.py
python src/profile_risk_segments.py
python src/build_reactivation_actions.py
```

For SQL:

```bash
python src/load_sqlite.py
sqlite3 data/processed/berka.sqlite < sql/01_source_profile.sql
```

## Technical stack

**Python · Pandas · NumPy · scikit-learn · SQL · SQLite · Git**

## Methodological notes

- Predictive features use only information available before each snapshot outcome window.
- The target is calculated from the following 90 days, as required for supervised learning.
- Model and targeting-policy selection use only the validation period.
- The final test period is opened once and is not used for further tuning.
- `account_id` is never used as a predictive feature.
- Campaign economics and reactivation actions are explicitly separated from observed model results because the source data contains no treatment experiment.

## Key result

> An out-of-time Gradient Boosting model achieved **0.9227 ROC-AUC** and **7.08x lift in the highest-risk 10%**, capturing **141 of 199 disengagement cases (70.85%)** while targeting only one-tenth of customers.
