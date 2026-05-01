from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from slope_warning.common.metrics import p95ae, time_block_folds


@dataclass(frozen=True)
class RegressionScore:
    rmse: float
    mae: float
    r2: float
    p95ae: float


def regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionScore:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return RegressionScore(
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        p95ae=float(p95ae(y_true, y_pred)),
    )


def time_block_model_scores(
    model_factory,
    x: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    k: int = 5,
    start: int = 0,
) -> dict[str, float]:
    x_df = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
    rows = []
    for test_idx in time_block_folds(len(y), k=k, start=start):
        train_idx = np.setdiff1d(np.arange(start, len(y)), test_idx, assume_unique=True)
        model = model_factory()
        model.fit(x_df.iloc[train_idx], y[train_idx])
        pred = model.predict(x_df.iloc[test_idx])
        rows.append(regression_score(y[test_idx], pred))
    return {
        "RMSE": float(np.mean([row.rmse for row in rows])),
        "MAE": float(np.mean([row.mae for row in rows])),
        "R2": float(np.mean([row.r2 for row in rows])),
        "P95AE": float(np.mean([row.p95ae for row in rows])),
    }


def monotonic_violations(values: np.ndarray, tol: float = 1e-9) -> int:
    return int(np.sum(np.diff(np.asarray(values, dtype=float)) < -tol))


def inverse_velocity_forecast(
    time: pd.Series,
    velocity: np.ndarray,
    stage: np.ndarray | None = None,
    window_steps: int = 144,
    min_velocity: float = 0.2,
    min_r2: float = 0.65,
) -> pd.DataFrame:
    velocity = np.asarray(velocity, dtype=float)
    stage = np.zeros(len(velocity), dtype=int) if stage is None else np.asarray(stage, dtype=int)
    inv = np.where(velocity > min_velocity, 1.0 / velocity, np.nan)
    rows = []
    x_all = np.arange(len(velocity), dtype=float) / 6.0
    for end in range(window_steps, len(velocity), 6):
        sl = slice(end - window_steps, end)
        y = inv[sl]
        x = x_all[sl] - x_all[end - window_steps]
        mask = np.isfinite(y)
        if mask.sum() < max(24, window_steps // 3):
            continue
        coef = np.polyfit(x[mask], y[mask], 1)
        pred = np.polyval(coef, x[mask])
        score = r2_score(y[mask], pred)
        if coef[0] >= 0 or score < min_r2:
            continue
        fail_h = -coef[1] / coef[0]
        current_h = x_all[end - 1] - x_all[end - window_steps]
        lead_h = fail_h - current_h
        if lead_h <= 0 or lead_h > 7 * 24:
            continue
        rows.append(
            {
                "row": int(end),
                "time": str(time.iloc[end - 1]),
                "stage": int(stage[end - 1]),
                "velocity_mm_h": float(velocity[end - 1]),
                "inverse_velocity": float(inv[end - 1]),
                "fit_slope": float(coef[0]),
                "fit_intercept": float(coef[1]),
                "fit_R2": float(score),
                "predicted_failure_time": str(time.iloc[end - 1] + pd.to_timedelta(lead_h, unit="h")),
                "lead_time_h": float(lead_h),
            }
        )
    columns = [
        "row",
        "time",
        "stage",
        "velocity_mm_h",
        "inverse_velocity",
        "fit_slope",
        "fit_intercept",
        "fit_R2",
        "predicted_failure_time",
        "lead_time_h",
    ]
    return pd.DataFrame(rows, columns=columns)


def git_diff_stat(repo: Path) -> str:
    result = subprocess.run(["git", "diff", "--stat"], cwd=repo, text=True, capture_output=True, check=False)
    return result.stdout.strip() or "(no working tree diff)"
