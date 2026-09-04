from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_targeting_metrics.csv"
)

# Illustrative campaign assumptions only.
# The Berka dataset does not contain campaign outcomes or customer lifetime value.
COST_PER_CONTACT = 2.0
SAVE_RATE_IF_TRULY_AT_RISK = 0.20


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Validation targeting metrics were not found. "
            "Run `python src/evaluate_thresholds.py` first."
        )

    df = pd.read_csv(METRICS_PATH)

    df["expected_saved_per_contact"] = (
        df["precision"] * SAVE_RATE_IF_TRULY_AT_RISK
    )

    df["break_even_retained_value"] = (
        COST_PER_CONTACT / df["expected_saved_per_contact"]
    )

    print("Illustrative intervention economics")
    print("-----------------------------------")
    print(f"Campaign cost per contact: {COST_PER_CONTACT:.2f} value units")
    print(
        "Assumed save rate among truly at-risk customers: "
        f"{SAVE_RATE_IF_TRULY_AT_RISK:.0%}"
    )
    print(
        "Note: campaign assumptions are illustrative; "
        "the source dataset contains no intervention outcomes.\n"
    )

    display = df[
        [
            "target_fraction",
            "precision",
            "recall",
            "lift",
            "break_even_retained_value",
        ]
    ].copy()

    print(
        display.to_string(
            index=False,
            formatters={
                "target_fraction": "{:.0%}".format,
                "precision": "{:.2%}".format,
                "recall": "{:.2%}".format,
                "lift": "{:.2f}x".format,
                "break_even_retained_value": "{:.2f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
