from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# On this macOS/NumPy environment, finite matrix multiplications emit spurious
# RuntimeWarnings ("divide by zero/overflow/invalid encountered in matmul").
# We verified separately that both inputs and outputs are finite, so suppress
# only this specific warning family rather than hiding RuntimeWarnings broadly.
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


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def top_fraction_metrics(y_true, y_score, fraction=0.10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_top = max(1, int(np.ceil(len(y_true) * fraction)))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    prevalence = y_true.mean()
    top_precision = y_true[top_idx].mean()
    lift = top_precision / prevalence if prevalence > 0 else np.nan
    return n_top, top_precision, lift


def main():
    print("Loading modelling dataset...")
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()

    feature_cols = [c for c in df.columns if c not in ID_COLS]

    for frame in (train, validation):
        frame["district_id"] = frame["district_id"].astype(str)

    regular_numeric_features = [
        c for c in feature_cols
        if c not in CATEGORICAL_FEATURES + HEAVY_TAIL_FEATURES
    ]

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    X_val = validation[feature_cols]
    y_val = validation[TARGET].astype(int)

    print(f"Train rows: {len(train):,} | positives: {y_train.sum():,} ({y_train.mean():.2%})")
    print(f"Validation rows: {len(validation):,} | positives: {y_val.sum():,} ({y_val.mean():.2%})")
    print(f"Regular numeric features: {len(regular_numeric_features)}")
    print(f"Heavy-tail features: {HEAVY_TAIL_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")

    heavy_tail_pipeline = Pipeline([
        ("signed_log1p", FunctionTransformer(signed_log1p, validate=False)),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), regular_numeric_features),
        ("heavy", heavy_tail_pipeline, HEAVY_TAIL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

    model = LogisticRegression(
        class_weight="balanced",
        C=0.1,
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    print("\nTraining logistic-regression baseline...")
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, val_prob)
    pr_auc = average_precision_score(y_val, val_prob)
    n_top, top_precision, lift = top_fraction_metrics(y_val, val_prob, fraction=0.10)

    print("\nValidation metrics")
    print("------------------")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"Baseline prevalence: {y_val.mean():.4f}")
    print(f"Top 10% rows:        {n_top:,}")
    print(f"Top 10% precision:   {top_precision:.4f}")
    print(f"Top 10% lift:        {lift:.2f}x")

    print("\nBaseline complete.")


if __name__ == "__main__":
    main()
