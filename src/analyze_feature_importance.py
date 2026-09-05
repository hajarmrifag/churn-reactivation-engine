from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "validation_feature_importance.csv"

TARGET = "target_disengaged"
ID_COLS = ["account_id", "snapshot_date", "split", TARGET]
CATEGORICAL_FEATURES = ["district_id", "frequency", "latest_card_type"]
HEAVY_TAIL_FEATURES = [
    "incoming_outgoing_ratio_90d",
    "amount_change_pct_30d",
]

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

    print("Training selected Gradient Boosting model on training period...")
    model.fit(X_train, y_train)

    print("Calculating permutation importance on validation period...")
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance.to_csv(OUTPUT_PATH, index=False)

    print("\nTop 15 validation permutation importances (PR-AUC decrease)")
    print("----------------------------------------------------------")
    print(
        importance.head(15).to_string(
            index=False,
            formatters={
                "importance_mean": "{:.6f}".format,
                "importance_std": "{:.6f}".format,
            },
        )
    )

    print(f"\nSaved importance table: {OUTPUT_PATH}")
    print(
        "Interpretation: larger positive values mean validation PR-AUC "
        "drops more when that feature is shuffled."
    )

if __name__ == "__main__":
    main()
