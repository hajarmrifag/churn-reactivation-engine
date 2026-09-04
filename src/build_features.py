from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "berka"
PROCESSED_DIR = ROOT / "data" / "processed"

SNAPSHOTS = pd.date_range("1997-01-01", "1998-09-01", freq="MS")
LOOKBACK_DAYS = 90
FUTURE_DAYS = 90
MIN_PRIOR_TRANSACTIONS = 4
SEASONAL_THRESHOLD_MULTIPLIER = 0.60

CUSTOMER_OPS = {
    "VYBER",
    "PREVOD NA UCET",
    "VYBER KARTOU",
    "VKLAD",
}


def parse_berka_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.zfill(6),
        format="%y%m%d",
        errors="raise",
    )


def load_data() -> dict[str, pd.DataFrame]:
    trans = pd.read_csv(
        RAW_DIR / "trans.csv",
        sep=";",
        low_memory=False,
    )
    account = pd.read_csv(RAW_DIR / "account.csv", sep=";")
    card = pd.read_csv(RAW_DIR / "card.csv", sep=";")
    disp = pd.read_csv(RAW_DIR / "disp.csv", sep=";")
    loan = pd.read_csv(RAW_DIR / "loan.csv", sep=";")

    trans["date"] = parse_berka_date(trans["date"])
    account["date"] = parse_berka_date(account["date"])
    loan["date"] = parse_berka_date(loan["date"])
    card["issued"] = pd.to_datetime(
        card["issued"],
        format="%y%m%d %H:%M:%S",
        errors="raise",
    )

    return {
        "trans": trans,
        "account": account,
        "card": card,
        "disp": disp,
        "loan": loan,
    }


def make_static_account_features(
    account: pd.DataFrame,
    disp: pd.DataFrame,
    card: pd.DataFrame,
    loan: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    static = account.loc[
        account["date"] < snapshot,
        ["account_id", "district_id", "frequency", "date"],
    ].copy()

    static = static.rename(columns={"date": "account_open_date"})
    static["account_age_days"] = (
        snapshot - static["account_open_date"]
    ).dt.days

    owner_disp = disp.loc[
        disp["type"].eq("OWNER"),
        ["disp_id", "account_id"],
    ].copy()

    cards_to_date = (
        card.merge(owner_disp, on="disp_id", how="inner")
        .loc[lambda x: x["issued"] < snapshot]
        .sort_values(["account_id", "issued"])
        .groupby("account_id", as_index=False)
        .agg(
            has_card=("card_id", "size"),
            latest_card_type=("type", "last"),
        )
    )
    cards_to_date["has_card"] = 1

    loans_to_date = (
        loan.loc[
            loan["date"] < snapshot,
            [
                "account_id",
                "amount",
                "duration",
                "payments",
            ],
        ]
        .sort_values("account_id")
        .groupby("account_id", as_index=False)
        .agg(
            has_loan=("amount", "size"),
            loan_amount=("amount", "max"),
            loan_duration_months=("duration", "max"),
            loan_payment=("payments", "max"),
        )
    )
    loans_to_date["has_loan"] = 1

    static = static.merge(
        cards_to_date[
            ["account_id", "has_card", "latest_card_type"]
        ],
        on="account_id",
        how="left",
    )
    static = static.merge(
        loans_to_date,
        on="account_id",
        how="left",
    )

    static["has_card"] = static["has_card"].fillna(0).astype(int)
    static["latest_card_type"] = static["latest_card_type"].fillna("none")
    static["has_loan"] = static["has_loan"].fillna(0).astype(int)
    static["loan_amount"] = static["loan_amount"].fillna(0.0)
    static["loan_duration_months"] = (
        static["loan_duration_months"].fillna(0).astype(int)
    )
    static["loan_payment"] = static["loan_payment"].fillna(0.0)

    return static


def aggregate_window(
    trans: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    suffix: str,
) -> pd.DataFrame:
    window = trans.loc[
        (trans["date"] >= start)
        & (trans["date"] < end)
    ].copy()

    if window.empty:
        return pd.DataFrame(columns=["account_id"])

    window["is_customer_op"] = window["operation"].isin(CUSTOMER_OPS)
    window["is_incoming"] = window["type"].eq("PRIJEM")
    window["is_outgoing"] = window["type"].isin(["VYDAJ", "VYBER"])
    window["is_cash_withdrawal"] = window["operation"].eq("VYBER")
    window["is_card_withdrawal"] = window["operation"].eq("VYBER KARTOU")
    window["is_outgoing_transfer"] = window["operation"].eq("PREVOD NA UCET")
    window["is_cash_deposit"] = window["operation"].eq("VKLAD")

    customer_window = window.loc[window["is_customer_op"]].copy()

    all_agg = (
        window.groupby("account_id", as_index=False)
        .agg(
            all_tx_count=("trans_id", "size"),
            total_amount=("amount", "sum"),
            mean_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            mean_balance=("balance", "mean"),
            min_balance=("balance", "min"),
            max_balance=("balance", "max"),
            last_balance=("balance", "last"),
            incoming_count=("is_incoming", "sum"),
            outgoing_count=("is_outgoing", "sum"),
            incoming_amount=(
                "amount",
                lambda s: s[
                    window.loc[s.index, "is_incoming"]
                ].sum(),
            ),
            outgoing_amount=(
                "amount",
                lambda s: s[
                    window.loc[s.index, "is_outgoing"]
                ].sum(),
            ),
        )
    )

    if customer_window.empty:
        customer_agg = pd.DataFrame(
            columns=[
                "account_id",
                "customer_tx_count",
                "active_days",
                "unique_operations",
                "cash_withdrawals",
                "card_withdrawals",
                "outgoing_transfers",
                "cash_deposits",
                "customer_amount",
                "last_customer_tx_date",
            ]
        )
    else:
        customer_agg = (
            customer_window.groupby("account_id", as_index=False)
            .agg(
                customer_tx_count=("trans_id", "size"),
                active_days=("date", lambda s: s.dt.normalize().nunique()),
                unique_operations=("operation", "nunique"),
                cash_withdrawals=("is_cash_withdrawal", "sum"),
                card_withdrawals=("is_card_withdrawal", "sum"),
                outgoing_transfers=("is_outgoing_transfer", "sum"),
                cash_deposits=("is_cash_deposit", "sum"),
                customer_amount=("amount", "sum"),
                last_customer_tx_date=("date", "max"),
            )
        )

    result = all_agg.merge(
        customer_agg,
        on="account_id",
        how="left",
    )

    result = result.rename(
        columns={
            col: f"{col}_{suffix}"
            for col in result.columns
            if col != "account_id"
        }
    )

    return result


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def build_snapshot(
    data: dict[str, pd.DataFrame],
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    trans = data["trans"]
    account = data["account"]

    past_90_start = snapshot - pd.Timedelta(days=90)
    past_30_start = snapshot - pd.Timedelta(days=30)
    past_60_start = snapshot - pd.Timedelta(days=60)
    future_end = snapshot + pd.Timedelta(days=90)

    qualifying_past = trans.loc[
        (trans["date"] >= past_90_start)
        & (trans["date"] < snapshot)
        & (trans["operation"].isin(CUSTOMER_OPS))
    ]

    past_counts = (
        qualifying_past.groupby("account_id")
        .size()
        .rename("customer_tx_count_90d")
    )

    eligible_ids = past_counts.loc[
        past_counts >= MIN_PRIOR_TRANSACTIONS
    ].index

    static = make_static_account_features(
        account,
        data["disp"],
        data["card"],
        data["loan"],
        snapshot,
    )
    static = static.loc[
        static["account_id"].isin(eligible_ids)
    ].copy()

    windows = [
        aggregate_window(
            trans,
            past_90_start,
            snapshot,
            "90d",
        ),
        aggregate_window(
            trans,
            past_30_start,
            snapshot,
            "30d",
        ),
        aggregate_window(
            trans,
            past_60_start,
            past_30_start,
            "prev30d",
        ),
        aggregate_window(
            trans,
            past_90_start,
            past_60_start,
            "prevprev30d",
        ),
    ]

    features = static
    for window_df in windows:
        features = features.merge(
            window_df,
            on="account_id",
            how="left",
        )

    numeric_cols = features.select_dtypes(
        include=["number"]
    ).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    date_cols = [
        col
        for col in features.columns
        if col.startswith("last_customer_tx_date_")
    ]

    for col in date_cols:
        days_name = col.replace(
            "last_customer_tx_date_",
            "recency_days_",
        )
        features[days_name] = (
            snapshot - pd.to_datetime(features[col])
        ).dt.days
        features[days_name] = (
            features[days_name]
            .fillna(999)
            .clip(lower=0)
        )

    features = features.drop(
        columns=date_cols + ["account_open_date"],
        errors="ignore",
    )

    features["tx_count_change_30d"] = (
        features["customer_tx_count_30d"]
        - features["customer_tx_count_prev30d"]
    )

    features["tx_count_change_pct_30d"] = safe_ratio(
        features["customer_tx_count_30d"]
        - features["customer_tx_count_prev30d"],
        features["customer_tx_count_prev30d"],
    )

    features["tx_count_recent_vs_90d_share"] = safe_ratio(
        features["customer_tx_count_30d"],
        features["customer_tx_count_90d"],
    )

    features["amount_change_pct_30d"] = safe_ratio(
        features["customer_amount_30d"]
        - features["customer_amount_prev30d"],
        features["customer_amount_prev30d"],
    )

    features["balance_change_30d"] = (
        features["last_balance_30d"]
        - features["last_balance_prev30d"]
    )

    features["incoming_outgoing_ratio_90d"] = safe_ratio(
        features["incoming_amount_90d"],
        features["outgoing_amount_90d"],
    )

    features["failed_like_balance_pressure"] = (
        features["min_balance_90d"] < 0
    ).astype(int)

    future = trans.loc[
        (trans["date"] >= snapshot)
        & (trans["date"] < future_end)
        & (trans["operation"].isin(CUSTOMER_OPS))
    ]

    future_counts = (
        future.groupby("account_id")
        .size()
        .rename("future_customer_tx_count")
    )

    label_frame = (
        past_counts.loc[eligible_ids]
        .to_frame()
        .join(future_counts, how="left")
        .fillna({"future_customer_tx_count": 0})
    )

    label_frame["future_past_ratio"] = (
        label_frame["future_customer_tx_count"]
        / label_frame["customer_tx_count_90d"]
    )

    seasonal_median = label_frame["future_past_ratio"].median()
    label_threshold = (
        seasonal_median * SEASONAL_THRESHOLD_MULTIPLIER
    )

    label_frame["target_disengaged"] = (
        label_frame["future_past_ratio"]
        <= label_threshold
    ).astype(int)

    labels = (
        label_frame[["target_disengaged"]]
        .reset_index()
    )

    features = features.merge(
        labels,
        on="account_id",
        how="inner",
    )

    features["snapshot_date"] = snapshot

    if snapshot < pd.Timestamp("1998-01-01"):
        split = "train"
    elif snapshot < pd.Timestamp("1998-07-01"):
        split = "validation"
    else:
        split = "test"

    features["split"] = split

    ordered_first = [
        "account_id",
        "snapshot_date",
        "split",
        "target_disengaged",
    ]
    remaining = [
        col
        for col in features.columns
        if col not in ordered_first
    ]

    return features[ordered_first + remaining]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Berka banking data...")
    data = load_data()

    snapshots = []

    for snapshot in SNAPSHOTS:
        frame = build_snapshot(data, snapshot)
        snapshots.append(frame)

        rate = frame["target_disengaged"].mean()
        print(
            f"{snapshot.date()} | "
            f"rows={len(frame):,} | "
            f"positive_rate={rate:.2%}"
        )

    dataset = pd.concat(
        snapshots,
        ignore_index=True,
    )

    output_path = (
        PROCESSED_DIR
        / "churn_model_dataset.csv"
    )
    dataset.to_csv(
        output_path,
        index=False,
    )

    print("\nDataset summary")
    print("=" * 40)
    print(f"Rows: {len(dataset):,}")
    print(f"Accounts: {dataset['account_id'].nunique():,}")
    print(f"Features: {dataset.shape[1] - 4:,}")

    split_summary = (
        dataset.groupby("split")
        .agg(
            rows=("target_disengaged", "size"),
            positives=("target_disengaged", "sum"),
            positive_rate=("target_disengaged", "mean"),
        )
    )

    print("\nTime-based splits:")
    print(split_summary.to_string(
        formatters={
            "positive_rate": lambda x: f"{x:.2%}"
        }
    ))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
