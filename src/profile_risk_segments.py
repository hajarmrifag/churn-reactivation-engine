from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_dataset.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_risk_segment_profile.csv"
)

TARGET = "target_disengaged"
ID_COLS = ["account_id", "snapshot_date", "split", TARGET]
CATEGORICAL_FEATURES = ["district_id", "frequency", "latest_card_type"]
HEAVY_TAIL_FEATURES = [
    "incoming_outgoing_ratio_90d",
    "amount_change_pct_30d",
]

PROFILE_FEATURES = [
    "cash_withdrawals_90d",
    "outgoing_transfers_30d",
    "recency_days_prevprev30d",
    "unique_operations_90d",
    "customer_amount_90d",
    "incoming_amount_30d",
    "all_tx_count_90d",
    "cash_deposits_90d",
    "balance_change_30d",
    "account_age_days",
]

TARGET_FRACTION = 0.10


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(feature_cols):
    regular_numeric_features = [
        c
        for c in feature_cols
        if c not in CATEGORICAL_FEATURES + HEAVY_TAIL_FEATURES
    ]

    heavy_tail_pipeline = Pipeline([
        ("signed_log1p", FunctionTransformer(signed_log1p, validate=False)),
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer([
        ("num", StandardScaler(), regular_numeric_features),
        ("heavy", heavy_tail_pipeline, HEAVY_TAIL_FEATURES),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
    ])


def main():
    print("Loading modelling dataset...")
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()

    for frame in (train, validation):
        frame["district_id"] = frame["district_id"].astype(str)

    feature_cols = [c for c in df.columns if c not in ID_COLS]

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)

    X_val = validation[feature_cols]
    y_val = validation[TARGET].astype(int)

    model = Pipeline([
        ("preprocess", build_preprocessor(feature_cols)),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("Training selected Gradient Boosting model...")
    model.fit(X_train, y_train)

    validation["risk_score"] = model.predict_proba(X_val)[:, 1]

    cutoff = validation["risk_score"].quantile(1 - TARGET_FRACTION)
    validation["risk_segment"] = np.where(
        validation["risk_score"] >= cutoff,
        "top_10pct_risk",
        "other_90pct",
    )

    high = validation[validation["risk_segment"] == "top_10pct_risk"]
    other = validation[validation["risk_segment"] == "other_90pct"]

    rows = []
    for feature in PROFILE_FEATURES:
        high_median = high[feature].median()
        other_median = other[feature].median()
        high_mean = high[feature].mean()
        other_mean = other[feature].mean()

        overall_std = validation[feature].std(ddof=0)
        standardized_mean_diff = (
            (high_mean - other_mean) / overall_std
            if overall_std > 0
            else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "high_risk_median": high_median,
                "other_median": other_median,
                "high_risk_mean": high_mean,
                "other_mean": other_mean,
                "standardized_mean_diff": standardized_mean_diff,
            }
        )

    profile = pd.DataFrame(rows)
    profile["abs_standardized_diff"] = profile["standardized_mean_diff"].abs()
    profile = profile.sort_values(
        "abs_standardized_diff",
        ascending=False,
    ).drop(columns="abs_standardized_diff")

    profile.to_csv(OUTPUT_PATH, index=False)

    print("\nValidation risk-segment profile")
    print("-------------------------------")
    print(f"High-risk rows: {len(high):,}")
    print(f"Other rows:     {len(other):,}")
    print(
        f"Observed disengagement rate, high-risk: "
        f"{high[TARGET].mean():.2%}"
    )
    print(
        f"Observed disengagement rate, other:     "
        f"{other[TARGET].mean():.2%}"
    )

    print("\nTop behavioural differences")
    print("---------------------------")
    print(
        profile.to_string(
            index=False,
            formatters={
                "high_risk_median": "{:.2f}".format,
                "other_median": "{:.2f}".format,
                "high_risk_mean": "{:.2f}".format,
                "other_mean": "{:.2f}".format,
                "standardized_mean_diff": "{:+.2f}".format,
            },
        )
    )

    print(f"\nSaved profile: {OUTPUT_PATH}")
    print(
        "Positive standardized differences mean the high-risk group has "
        "higher values; negative differences mean lower values."
    )


if __name__ == "__main__":
    main()
