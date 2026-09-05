from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_dataset.csv"

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


def top_fraction_metrics(y_true, scores, fraction):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n_targeted = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(scores)[::-1]
    idx = order[:n_targeted]

    positives_total = y_true.sum()
    positives_targeted = y_true[idx].sum()
    prevalence = y_true.mean()

    precision = positives_targeted / n_targeted
    recall = positives_targeted / positives_total if positives_total else np.nan
    lift = precision / prevalence if prevalence else np.nan

    return {
        "targeted_rows": n_targeted,
        "positives_captured": int(positives_targeted),
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def main():
    print("Loading modelling dataset...")
    df = pd.read_csv(DATA_PATH)

    development = df[df["split"].isin(["train", "validation"])].copy()
    test = df[df["split"] == "test"].copy()

    for frame in (development, test):
        frame["district_id"] = frame["district_id"].astype(str)

    feature_cols = [c for c in df.columns if c not in ID_COLS]

    X_dev = development[feature_cols]
    y_dev = development[TARGET].astype(int)

    X_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    print(
        f"Development: {len(development):,} rows | "
        f"{y_dev.sum():,} positives ({y_dev.mean():.2%})"
    )
    print(
        f"Test: {len(test):,} rows | "
        f"{y_test.sum():,} positives ({y_test.mean():.2%})"
    )

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

    print("\nTraining final Gradient Boosting model on train + validation...")
    model.fit(X_dev, y_dev)

    test_scores = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    targeting = top_fraction_metrics(y_test, test_scores, TARGET_FRACTION)

    print("\nLocked test results")
    print("-------------------")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"Test prevalence:    {y_test.mean():.4f}")
    print(f"Targeting policy:   top {TARGET_FRACTION:.0%} by risk score")
    print(f"Targeted rows:      {targeting['targeted_rows']:,}")
    print(f"Cases captured:     {targeting['positives_captured']:,} / {int(y_test.sum()):,}")
    print(f"Top-10% precision:  {targeting['precision']:.4f}")
    print(f"Top-10% recall:     {targeting['recall']:.4f}")
    print(f"Top-10% lift:       {targeting['lift']:.2f}x")

    print(
        "\nImportant: treat these as final holdout results. "
        "Do not tune the model or targeting rule against this test set."
    )


if __name__ == "__main__":
    main()
