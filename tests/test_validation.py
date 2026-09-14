from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_purged import purged_training_rows, signed_log1p, top_fraction_metrics
from evaluate_thresholds import evaluate_fraction


def test_top_fraction_metrics_selects_highest_scores():
    labels = [0, 1, 0, 1, 0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3]

    result = top_fraction_metrics(labels, scores, fraction=0.4)

    assert result["targeted_rows"] == 2
    assert result["positives_captured"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["lift"] == 2.5


def test_targeting_metrics_handle_a_population_without_positives():
    result = evaluate_fraction([0, 0, 0], [0.3, 0.2, 0.1], fraction=0.1)

    assert result["targeted_rows"] == 1
    assert result["positives_captured"] == 0
    assert result["precision"] == 0
    assert np.isnan(result["recall"])
    assert np.isnan(result["lift"])


def test_signed_log_transform_preserves_sign_and_compresses_magnitude():
    transformed = signed_log1p(np.array([-99.0, 0.0, 99.0]))

    np.testing.assert_allclose(
        transformed,
        np.array([-np.log(100), 0.0, np.log(100)]),
    )


def test_temporal_purge_excludes_overlapping_outcome_windows():
    frame = pd.DataFrame(
        {
            "snapshot_date": ["1997-09-30", "1997-10-04", "1998-01-01"],
            "split": ["train", "train", "validation"],
        }
    )

    result = purged_training_rows(
        frame,
        training_splits=["train"],
        evaluation_start="1998-01-01",
    )

    assert result["snapshot_date"].tolist() == [pd.Timestamp("1997-09-30")]
    assert (result["label_end"] <= pd.Timestamp("1998-01-01")).all()
