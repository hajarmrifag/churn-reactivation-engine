# Model card: behavioural disengagement ranking

## Summary

This project ranks active bank account-month observations by the likelihood of a severe, seasonally adjusted decline in customer-initiated transaction activity over the following 90 days. It is a portfolio research system, not a deployed bank model.

## Intended use

- Explore whether historical activity can support early customer-engagement analysis.
- Rank observations for analyst review under an explicitly chosen contact capacity.
- Demonstrate leakage-aware temporal validation and decision-oriented evaluation.
- Support scenario analysis in which all intervention costs and effects remain visible assumptions.

## Out-of-scope and prohibited uses

- Credit, fraud, pricing, eligibility, account closure, or adverse-action decisions.
- Fully automated customer contact or treatment assignment.
- Interpreting behavioural disengagement as confirmed churn or financial distress.
- Applying the reported metrics to current customers, other markets, or other time periods without new validation.

## Data lineage

The analysis uses the anonymized Berka / PKDD'99 Czech financial dataset. Raw CSV files are not committed. The feature pipeline produces monthly snapshots from January 1997 through September 1998 using only information available at each snapshot. The outcome is derived from the subsequent 90 days of activity; it is not a source-system label.

## Evaluation design

- Chronological train, validation, and locked-test periods replace a random split.
- A stricter purged evaluation removes training rows whose 90-day label windows overlap the next evaluation period.
- Because the positive class is rare, PR-AUC, precision, recall, and lift accompany ROC-AUC.
- Targeting capacity is chosen on validation data; the test period is not used to tune the policy.
- Results refer to account-month observations, so one account may occur in multiple snapshots.

## Known limitations and risks

- The dataset is historical, geographically narrow, and unlikely to represent a modern customer population.
- The target is a research proxy and may capture seasonality, changing needs, or ordinary account behaviour rather than intent to leave.
- No protected-characteristic fairness assessment is claimed because an appropriate modern population and deployment context are absent.
- Feature importance describes predictive association, not causality.
- The data contains no retention intervention, customer lifetime value, or observed treatment effect. Campaign economics are illustrative.
- Distribution shift, missing upstream data, policy changes, and feedback loops would all affect a deployed system.

## Human oversight

A real implementation should provide scores and reason codes to trained analysts, require a documented review before customer action, provide a route to correct underlying data, and log the model version, feature timestamp, score, decision, reviewer, and eventual outcome. A model score alone should never trigger an adverse action.

## Monitoring plan for a real deployment

Before launch, establish population and segment baselines. Then monitor feature missingness and drift, score distributions, calibration, precision and recall after labels mature, contact rates, customer outcomes, and performance across legally and operationally relevant groups. Define alert thresholds, an owner, a rollback path, and scheduled revalidation. Version the data contract, code, model artifact, threshold, and approval record together.

## Reproducibility boundary

Unit tests and CI validate source-independent metric and leakage-control logic. Reproducing model results requires obtaining the source dataset separately and following the repository pipeline. This distinction is intentional: the project does not fabricate a full CI retraining claim when the raw data is not part of the repository.
