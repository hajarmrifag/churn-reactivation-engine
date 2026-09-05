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
    / "validation_reactivation_actions.csv"
)

TARGET = "target_disengaged"
ID_COLS = ["account_id", "snapshot_date", "split", TARGET]
CATEGORICAL_FEATURES = ["district_id", "frequency", "latest_card_type"]
HEAVY_TAIL_FEATURES = [
    "incoming_outgoing_ratio_90d",
    "amount_change_pct_30d",
]
TARGET_FRACTION = 0.10


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(feature_cols):
    regular_numeric_features = [
        c for c in feature_cols
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


def assign_action(row):
    """
    Assign a hypothesis-driven reactivation action.

    These are operational hypotheses based on observed behaviour patterns,
    not experimentally proven treatment effects.
    """
    if row["balance_change_30d"] < 0 and row["cash_deposits_90d"] <= 1:
        return "account_health_check_in"

    if row["unique_operations_90d"] <= 2 and row["outgoing_transfers_30d"] == 0:
        return "payments_reengagement"

    if row["cash_withdrawals_90d"] >= 12 and row["cash_deposits_90d"] <= 1:
        return "deposit_reengagement"

    if row["unique_operations_90d"] <= 2:
        return "product_usage_reengagement"

    return "general_reactivation"


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
    high_risk = validation[
        validation["risk_score"] >= cutoff
    ].copy()

    high_risk["reactivation_action"] = high_risk.apply(
        assign_action,
        axis=1,
    )

    action_summary = (
        high_risk.groupby("reactivation_action")
        .agg(
            customers=("account_id", "size"),
            observed_disengagement_rate=(TARGET, "mean"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
        .sort_values("customers", ascending=False)
    )

    output_cols = [
        "account_id",
        "snapshot_date",
        "risk_score",
        TARGET,
        "reactivation_action",
        "cash_withdrawals_90d",
        "cash_deposits_90d",
        "outgoing_transfers_30d",
        "unique_operations_90d",
        "balance_change_30d",
    ]
    high_risk[output_cols].to_csv(OUTPUT_PATH, index=False)

    print("\nHigh-risk reactivation action mix")
    print("---------------------------------")
    print(
        action_summary.to_string(
            index=False,
            formatters={
                "observed_disengagement_rate": "{:.2%}".format,
                "mean_risk_score": "{:.4f}".format,
            },
        )
    )

    print(f"\nHigh-risk rows assigned: {len(high_risk):,}")
    print(f"Saved action file: {OUTPUT_PATH}")
    print(
        "\nImportant: action labels are hypotheses for campaign design. "
        "They are not causal treatment recommendations because the dataset "
        "contains no intervention experiment."
    )


if __name__ == "__main__":
    main()
