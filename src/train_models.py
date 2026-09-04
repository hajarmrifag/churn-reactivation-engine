from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

    return top_precision, lift


def build_preprocessor(feature_cols):
    regular_numeric_features = [
        c
        for c in feature_cols
        if c not in CATEGORICAL_FEATURES + HEAVY_TAIL_FEATURES
    ]

    heavy_tail_pipeline = Pipeline(
        steps=[
            ("signed_log1p", FunctionTransformer(signed_log1p, validate=False)),
            ("scale", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), regular_numeric_features),
            ("heavy", heavy_tail_pipeline, HEAVY_TAIL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val):
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    val_prob = pipeline.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, val_prob)
    pr_auc = average_precision_score(y_val, val_prob)
    top_precision, lift = top_fraction_metrics(y_val, val_prob, fraction=0.10)

    print(
        f"{name}: ROC-AUC={roc_auc:.4f} | "
        f"PR-AUC={pr_auc:.4f} | "
        f"Top-10% precision={top_precision:.4f} | "
        f"Top-10% lift={lift:.2f}x"
    )

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "top_10_precision": top_precision,
        "top_10_lift": lift,
    }


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

    print(
        f"Train: {len(train):,} rows, {y_train.sum():,} positives "
        f"({y_train.mean():.2%})"
    )
    print(
        f"Validation: {len(validation):,} rows, {y_val.sum():,} positives "
        f"({y_val.mean():.2%})"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_cols)),
                ("model", model),
            ]
        )

        results.append(
            evaluate_model(
                name,
                pipeline,
                X_train,
                y_train,
                X_val,
                y_val,
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        ["pr_auc", "top_10_lift"],
        ascending=False,
    )

    print("\nValidation comparison")
    print("---------------------")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.4f}".format,
                "pr_auc": "{:.4f}".format,
                "top_10_precision": "{:.4f}".format,
                "top_10_lift": "{:.2f}x".format,
            },
        )
    )

    print("\nBest model by PR-AUC:", results_df.iloc[0]["model"])


if __name__ == "__main__":
    main()
