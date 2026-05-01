from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.nanmean(err * err)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.nanmean(np.abs(err)))


def p95ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.nanpercentile(np.abs(err), 95))


def maxae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.nanmax(np.abs(err)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def time_block_folds(n: int, k: int = 5, start: int = 0) -> list[np.ndarray]:
    idx = np.arange(start, n)
    return [fold for fold in np.array_split(idx, k) if len(fold) > 0]

