from __future__ import annotations

import numpy as np
import pandas as pd

from slope_warning.common.io import read_excel, write_csv, write_json
from slope_warning.common.metrics import mae, maxae, p95ae, rmse, time_block_folds
from slope_warning.config import ATTACHMENTS, MODEL_DIR, TABLE_DIR


TARGET_VALUES = np.array([7.132, 18.526, 84.337, 123.554, 167.667], dtype=float)


def huber_affine_fit(x: np.ndarray, y: np.ndarray, delta_scale: float = 1.345, max_iter: int = 80) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - design @ beta
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(residual)
        if scale <= 1e-12:
            break
        delta = delta_scale * scale
        weights = np.ones_like(residual)
        mask = np.abs(residual) > delta
        weights[mask] = delta / np.abs(residual[mask])
        w_design = design * np.sqrt(weights[:, None])
        w_y = y * np.sqrt(weights)
        next_beta = np.linalg.lstsq(w_design, w_y, rcond=None)[0]
        if np.linalg.norm(next_beta - beta) < 1e-12:
            beta = next_beta
            break
        beta = next_beta
    return beta


def run() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q1"])
    x = df["数据A_光纤位移计数据_mm"].to_numpy(float)
    y = df["数据B_振弦式位移计数据_mm"].to_numpy(float)
    beta = huber_affine_fit(x, y)
    pred = beta[0] + beta[1] * x

    rows = []
    folds = time_block_folds(len(x), 5)
    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(x)), test_idx, assume_unique=True)
        fold_beta = huber_affine_fit(x[train_idx], y[train_idx])
        fold_pred = fold_beta[0] + fold_beta[1] * x[test_idx]
        rows.append(
            {
                "fold": fold_id,
                "beta0": fold_beta[0],
                "beta1": fold_beta[1],
                "RMSE_mm": rmse(y[test_idx], fold_pred),
                "MAE_mm": mae(y[test_idx], fold_pred),
                "P95AE_mm": p95ae(y[test_idx], fold_pred),
                "MaxAE_mm": maxae(y[test_idx], fold_pred),
            }
        )

    cv_df = pd.DataFrame(rows)
    table_df = pd.DataFrame(
        {
            "校正前数据x": TARGET_VALUES,
            "校正后数据y": beta[0] + beta[1] * TARGET_VALUES,
        }
    )
    model_summary = {
        "model": "Huber robust affine calibration: y = beta0 + beta1*x",
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "train_RMSE_mm": rmse(y, pred),
        "train_MAE_mm": mae(y, pred),
        "train_P95AE_mm": p95ae(y, pred),
        "train_MaxAE_mm": maxae(y, pred),
        "cv_mean": {
            "RMSE_mm": float(cv_df["RMSE_mm"].mean()),
            "MAE_mm": float(cv_df["MAE_mm"].mean()),
            "P95AE_mm": float(cv_df["P95AE_mm"].mean()),
            "MaxAE_mm": float(cv_df["MaxAE_mm"].mean()),
        },
        "table_1_1": [
            {"x": float(row["校正前数据x"]), "y": float(row["校正后数据y"])}
            for _, row in table_df.iterrows()
        ],
    }

    write_csv(cv_df, TABLE_DIR / "q1_cross_validation.csv")
    write_csv(table_df, TABLE_DIR / "q1_table_1_1.csv")
    write_json(model_summary, MODEL_DIR / "q1_model_summary.json")
    return model_summary

