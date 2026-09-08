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
OUTCOME_DAYS = 90

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


def build_model(feature_cols):
    return Pipeline([
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
def purged_training_rows(df, training_splits, evaluation_start):
    evaluation_start = pd.Timestamp(evaluation_start)

    training = df[
        df["split"].isin(training_splits)
    ].copy()

    training["snapshot_date"] = pd.to_datetime(
        training["snapshot_date"]
    )

    training["label_end"] = (
        training["snapshot_date"]
        + pd.to_timedelta(
            OUTCOME_DAYS,
            unit="D",
        )
    )

    return training[
        training["label_end"] <= evaluation_start
    ].copy()
def evaluate_split(df, training_splits, evaluation_split, evaluation_start):
    training = purged_training_rows(
        df,
        training_splits,
        evaluation_start,
    )

    evaluation = df[
        df["split"] == evaluation_split
    ].copy()

    training["district_id"] = (
        training["district_id"].astype(str)
    )
    evaluation["district_id"] = (
        evaluation["district_id"].astype(str)
    )

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLS
    ]

    X_train = training[feature_cols]
    y_train = training[TARGET].astype(int)

    X_eval = evaluation[feature_cols]
    y_eval = evaluation[TARGET].astype(int)

    model = build_model(feature_cols)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_eval)[:, 1]

    roc_auc = roc_auc_score(y_eval, scores)
    pr_auc = average_precision_score(y_eval, scores)
    targeting = top_fraction_metrics(
        y_eval,
        scores,
        TARGET_FRACTION,
    )

    return {
        "training_rows": len(training),
        "last_train_snapshot": training["snapshot_date"].max(),
        "evaluation_rows": len(evaluation),
        "positives": int(y_eval.sum()),
        "prevalence": y_eval.mean(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        **targeting,
    }
def main():
    print("Loading modelling dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["snapshot_date"],
    )

    validation = evaluate_split(
        df=df,
        training_splits=["train"],
        evaluation_split="validation",
        evaluation_start="1998-01-01",
    )

    test = evaluate_split(
        df=df,
        training_splits=["train", "validation"],
        evaluation_split="test",
        evaluation_start="1998-07-01",
    )

    print("\nPurged validation results")
    print("-------------------------")
    print(f"Training rows:       {validation['training_rows']:,}")
    print(f"Last train snapshot: {validation['last_train_snapshot'].date()}")
    print(f"Evaluation rows:     {validation['evaluation_rows']:,}")
    print(f"ROC-AUC:             {validation['roc_auc']:.4f}")
    print(f"PR-AUC:              {validation['pr_auc']:.4f}")
    print(f"Top-10% precision:   {validation['precision']:.4f}")
    print(f"Top-10% recall:      {validation['recall']:.4f}")
    print(f"Top-10% lift:        {validation['lift']:.2f}x")
    print(
        f"Cases captured:      "
        f"{validation['positives_captured']} / {validation['positives']}"
    )

    print("\nPurged locked-test results")
    print("--------------------------")
    print(f"Training rows:       {test['training_rows']:,}")
    print(f"Last train snapshot: {test['last_train_snapshot'].date()}")
    print(f"Evaluation rows:     {test['evaluation_rows']:,}")
    print(f"ROC-AUC:             {test['roc_auc']:.4f}")
    print(f"PR-AUC:              {test['pr_auc']:.4f}")
    print(f"Top-10% precision:   {test['precision']:.4f}")
    print(f"Top-10% recall:      {test['recall']:.4f}")
    print(f"Top-10% lift:        {test['lift']:.2f}x")
    print(
        f"Cases captured:      "
        f"{test['positives_captured']} / {test['positives']}"
    )


if __name__ == "__main__":
    main()
