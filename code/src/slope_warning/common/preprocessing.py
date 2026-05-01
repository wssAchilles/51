from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def rolling_median(values: np.ndarray, window: int, center: bool = False, min_periods: int | None = None) -> np.ndarray:
    if min_periods is None:
        min_periods = max(1, window // 3)
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=window, center=center, min_periods=min_periods)
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, dtype="float64").rolling(window=window, min_periods=1).sum().to_numpy()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, dtype="float64").rolling(window=window, min_periods=1).mean().to_numpy()


def robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.std(arr)
    return float(scale if scale > 1e-12 else 1.0)


def hampel_flags(values: np.ndarray, window: int = 37, threshold: float = 4.0, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    med = series.rolling(window=window, center=center, min_periods=max(3, window // 3)).median()
    resid = series - med
    mad = resid.abs().rolling(window=window, center=center, min_periods=max(3, window // 3)).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    score = (resid.abs() / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    flags = score.to_numpy() > threshold
    return flags, med.bfill().ffill().to_numpy(), score.to_numpy()


def hampel_replace(values: np.ndarray, window: int = 37, threshold: float = 4.0, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    flags, med, _ = hampel_flags(values, window=window, threshold=threshold, center=center)
    out = pd.Series(values, dtype="float64").interpolate(limit_direction="both").to_numpy().copy()
    out[flags] = med[flags]
    return out, flags


def sparse_series_fill(values: np.ndarray, integer: bool = False) -> np.ndarray:
    """Fill sparse event series without manufacturing smooth rainfall/event pulses."""
    s = pd.Series(values, dtype="float64")
    local = s.rolling(window=13, center=True, min_periods=1).median()
    filled = s.fillna(local).fillna(0.0).clip(lower=0)
    arr = filled.to_numpy()
    if integer:
        arr = np.rint(arr)
    return arr


def kalman_smooth_fill(values: np.ndarray, nonnegative: bool = False) -> np.ndarray:
    """Local-linear state-space smoother with interpolation fallback."""
    series = pd.Series(values, dtype="float64")
    if series.notna().sum() < 5:
        filled = series.interpolate(limit_direction="both").fillna(series.median()).fillna(0.0).to_numpy()
        return np.maximum(filled, 0.0) if nonnegative else filled

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(series.to_numpy(), level="local linear trend")
            result = model.fit(disp=False, maxiter=100)
        smoothed = result.smoothed_state[0]
        filled = series.to_numpy().copy()
        missing = ~np.isfinite(filled)
        filled[missing] = smoothed[missing]
        # Keep observed points intact, but use a light robust smoother for isolated sensor noise later.
    except Exception:
        filled = series.interpolate(method="linear", limit_direction="both").fillna(series.median()).to_numpy()
    if nonnegative:
        filled = np.maximum(filled, 0.0)
    return filled


def sparse_outlier_flags(values: np.ndarray, quantile: float = 0.995, min_threshold: float | None = None) -> np.ndarray:
    """Flag only extreme sparse-driver values; rainfall and event bursts are usually real signals."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    nonzero = finite[finite > 0]
    if nonzero.size < 8:
        return np.zeros_like(arr, dtype=bool)
    logv = np.log1p(nonzero)
    med = np.median(logv)
    scale = robust_scale(logv)
    robust_limit = np.expm1(med + 5.0 * scale)
    quantile_limit = np.quantile(nonzero, quantile)
    limit = max(robust_limit, quantile_limit)
    if min_threshold is not None:
        limit = max(limit, min_threshold)
    return np.isfinite(arr) & (arr > limit)
