from __future__ import annotations

import numpy as np
import pandas as pd

from slope_warning.common.metrics import time_block_folds
from slope_warning.common.preprocessing import hampel_replace, rolling_median
from slope_warning.common.segmentation import two_breaks_constant_mean
from slope_warning.questions.q1_calibration import huber_affine_fit
from slope_warning.questions.q4_prediction import _fit_stage_baselines, _baseline_values, _stage_from_breaks
from slope_warning.questions.q5_warning import _warning_thresholds


def test_time_block_folds_are_ordered_and_complete() -> None:
    folds = time_block_folds(11, k=5, start=1)
    merged = np.concatenate(folds)
    assert merged.tolist() == list(range(1, 11))
    assert all(np.all(np.diff(fold) == 1) for fold in folds)


def test_hampel_replace_suppresses_single_spike() -> None:
    values = np.ones(41)
    values[20] = 100.0
    cleaned, flags = hampel_replace(values, window=11, threshold=3.0, center=True)
    assert flags[20]
    assert cleaned[20] == 1.0


def test_rolling_median_preserves_length_and_fills_edges() -> None:
    values = np.array([np.nan, 1.0, 2.0, 100.0, 3.0])
    smoothed = rolling_median(values, window=3, center=True, min_periods=1)
    assert len(smoothed) == len(values)
    assert np.isfinite(smoothed).all()


def test_huber_affine_fit_is_robust_to_large_outlier() -> None:
    x = np.arange(30, dtype=float)
    y = 2.0 + 3.0 * x
    y[-1] += 500.0
    beta = huber_affine_fit(x, y, delta_scale=1.345)
    assert abs(beta[1] - 3.0) < 0.2


def test_two_breaks_constant_mean_recovers_three_levels() -> None:
    values = np.r_[np.zeros(20), np.ones(20) * 5, np.ones(20) * 12]
    b1, b2, _ = two_breaks_constant_mean(values, min_len=10, step=2, refine=4)
    assert abs(b1 - 20) <= 2
    assert abs(b2 - 40) <= 2


def test_q4_stage_baseline_is_monotone() -> None:
    y = np.r_[np.linspace(0, 10, 20), np.linspace(10, 40, 20), np.linspace(40, 100, 20)]
    stage = _stage_from_breaks(len(y), 20, 40)
    baselines = _fit_stage_baselines(y, stage)
    baseline = _baseline_values(stage, baselines)
    assert np.all(np.diff(baseline) >= -1e-9)


def test_q5_warning_thresholds_are_strictly_increasing() -> None:
    time = pd.date_range("2026-01-01", periods=180, freq="10min")
    velocity = np.r_[np.ones(60) * 1.0, np.ones(60) * 4.0, np.ones(60) * 10.0]
    stage = np.r_[np.ones(60), np.ones(60) * 2, np.ones(60) * 3].astype(int)
    thresholds, _, _ = _warning_thresholds(pd.Series(time), velocity, stage)
    values = thresholds["velocity_threshold_mm_h"].to_numpy(float)
    assert np.all(np.diff(values) > 0)
