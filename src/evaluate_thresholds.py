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
    / "validation_targeting_metrics.csv"
)

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


def evaluate_fraction(y_true, scores, fraction):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n = len(y_true)
    n_targeted = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(scores)[::-1]
    idx = order[:n_targeted]

    positives_total = y_true.sum()
    positives_targeted = y_true[idx].sum()

    precision = positives_targeted / n_targeted
    recall = positives_targeted / positives_total if positives_total else np.nan
    prevalence = y_true.mean()
    lift = precision / prevalence if prevalence else np.nan
    threshold = scores[idx[-1]]

    return {
        "target_fraction": fraction,
        "targeted_rows": n_targeted,
        "score_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "positives_captured": int(positives_targeted),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    for frame in (train, val):
        frame["district_id"] = frame["district_id"].astype(str)

    feature_cols = [c for c in df.columns if c not in ID_COLS]

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)

    X_val = val[feature_cols]
    y_val = val[TARGET].astype(int)

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
    scores = model.predict_proba(X_val)[:, 1]

    fractions = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    result = pd.DataFrame(
        [evaluate_fraction(y_val, scores, f) for f in fractions]
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("\nValidation targeting trade-offs")
    print("-------------------------------")
    print(
        result.to_string(
            index=False,
            formatters={
                "target_fraction": "{:.0%}".format,
                "score_threshold": "{:.4f}".format,
                "precision": "{:.2%}".format,
                "recall": "{:.2%}".format,
                "lift": "{:.2f}x".format,
            },
        )
    )

    print(f"\nValidation prevalence: {y_val.mean():.2%}")
    print(f"Validation positives: {int(y_val.sum()):,}")
    print(f"Saved metrics: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
