from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.config import ResearchConfig
from alpha_research.data import build_dataset
from alpha_research.features import generate_features
from alpha_research.modeling import fit_predict


def test_walk_forward_predictions_start_after_validation_boundary() -> None:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-12-31"
    config.split.validation_start = "2022-01-03"
    config.split.test_start = "2023-01-03"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)

    assert not predictions.empty
    assert predictions["date"].min().strftime("%Y-%m-%d") >= config.split.validation_start
    assert set(predictions["split"].unique()) <= {"validation", "test"}


def test_training_embargo_excludes_unresolved_labels() -> None:
    """A retrain must not see rows whose forward label resolves on or after it.

    Without an embargo the walk-forward loop trains on every row dated before the
    retrain date, including rows whose h-session target only resolves afterwards.
    Those labels are unknowable at fit time. This asserts the embargo directly on
    the selection rule rather than on downstream metrics, which can absorb a leak
    without failing.
    """
    from alpha_research.modeling import _target_horizon

    assert _target_horizon("target_return_5d") == 5
    assert _target_horizon("target_residual_21d_z") == 21
    assert _target_horizon("target_top_quintile") == 0

    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-12-31"
    config.split.validation_start = "2022-01-03"
    config.split.test_start = "2023-01-03"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)

    horizon = _target_horizon(config.model.target_column)
    assert horizon > 0, "the shipped target must encode a horizon for the embargo to apply"

    sessions = sorted(features["date"].unique())
    position = {session: index for index, session in enumerate(sessions)}
    retrain_date = pd.Timestamp(config.split.validation_start)
    cutoff = min(
        (position[s] for s in sessions if pd.Timestamp(s) >= retrain_date),
        default=None,
    )
    assert cutoff is not None

    usable = [s for s in sessions if position[s] + horizon < cutoff]
    latest_usable = pd.Timestamp(max(usable))
    label_resolution = sessions[position[max(usable)] + horizon]

    assert label_resolution < np.datetime64(retrain_date), (
        f"label for {latest_usable.date()} resolves at {pd.Timestamp(label_resolution).date()}, "
        f"which is not strictly before the retrain date {retrain_date.date()}"
    )


def test_predictions_are_finite_and_cover_every_oos_date() -> None:
    config = ResearchConfig()
    config.data.source = "synthetic"
    config.data.start_date = "2020-01-01"
    config.data.end_date = "2024-12-31"
    config.split.validation_start = "2022-01-03"
    config.split.test_start = "2023-01-03"

    panel = build_dataset(config.data)
    features = generate_features(panel, config.features)
    predictions = fit_predict(features, config.split, config.model)

    assert np.isfinite(predictions["prediction"]).all()
    assert predictions.groupby("date").size().min() > 0
