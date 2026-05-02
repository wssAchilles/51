from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slope_warning.common.diagnostics import monotonic_violations
from slope_warning.common.features import make_disturbance_features
from slope_warning.common.io import read_excel, write_csv
from slope_warning.common.metrics import time_block_folds
from slope_warning.common.segmentation import two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, AUDIT_CONFIG, AWARD_DIR, TABLE_DIR
from slope_warning.questions import q4_prediction


def _training_state() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    train = read_excel(ATTACHMENTS["q4"], sheet_name="训练集")
    train["时间"] = pd.to_datetime(train["时间"])
    y = train["表面位移_mm"].to_numpy(float)
    t_hours = (train["时间"] - train["时间"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    b1, b2, _ = two_breaks_piecewise_linear(t_hours, y, min_len=500, step=20, refine=120)
    stage = q4_prediction._stage_from_breaks(len(train), b1, b2)
    tau = q4_prediction._stage_tau(stage)
    baselines = q4_prediction._fit_stage_baselines(y, stage)
    baseline = q4_prediction._baseline_values(stage, baselines)
    residual = y - baseline
    features, _ = make_disturbance_features(train, stage=stage, tau=tau)
    return train, y, stage, baseline, residual, features


def _stage_residual_cv(
    features: pd.DataFrame,
    y: np.ndarray,
    stage: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    shrinkage: float,
) -> dict[str, float]:
    cv_pred = np.full_like(y, np.nan, dtype=float)
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        for local_test in time_block_folds(len(idx), k=3):
            test_idx = idx[local_test]
            train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
            model = q4_prediction._gbdt()
            model.fit(features.iloc[train_idx], residual[train_idx])
            raw_residual = model.predict(features.iloc[test_idx])
            cv_pred[test_idx] = baseline[test_idx] + shrinkage * raw_residual
    mask = np.isfinite(cv_pred)
    projected = q4_prediction._project_stagewise_monotone(cv_pred, lower_bound=baseline, stage=stage)
    projected_mask = np.isfinite(projected)
    pred = projected[projected_mask]
    raw_pred = cv_pred[mask]
    truth = y[mask]
    projected_truth = y[projected_mask]
    return {
        "shrinkage": shrinkage,
        "raw_RMSE_mm": float(mean_squared_error(truth, raw_pred) ** 0.5),
        "raw_MAE_mm": float(mean_absolute_error(truth, raw_pred)),
        "raw_monotonic_violations": monotonic_violations(raw_pred),
        "projected_RMSE_mm": float(mean_squared_error(projected_truth, pred) ** 0.5),
        "projected_MAE_mm": float(mean_absolute_error(projected_truth, pred)),
        "projected_monotonic_violations": monotonic_violations(pred),
        "RMSE_mm": float(mean_squared_error(projected_truth, pred) ** 0.5),
        "MAE_mm": float(mean_absolute_error(projected_truth, pred)),
        "monotonic_violations": monotonic_violations(pred),
    }


def _prediction_interval_coverage(stage: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
    rows = []
    total_count = 0
    covered_count = 0
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        stage_resid = residual[idx]
        width = max(1.96 * np.std(stage_resid), np.percentile(np.abs(stage_resid), 90))
        covered = int(np.sum(np.abs(stage_resid) <= width))
        rows.append(
            {
                "stage": int(label),
                "sample_count": int(len(idx)),
                "interval_half_width_mm": float(width),
                "empirical_coverage": float(covered / max(len(idx), 1)),
                "mean_abs_residual_mm": float(np.mean(np.abs(stage_resid))),
                "p90_abs_residual_mm": float(np.percentile(np.abs(stage_resid), 90)),
            }
        )
        total_count += int(len(idx))
        covered_count += covered
    overall = pd.DataFrame(rows)
    rows.append(
        {
            "stage": "overall",
            "sample_count": total_count,
            "interval_half_width_mm": float(overall["interval_half_width_mm"].mean()),
            "empirical_coverage": float(covered_count / max(total_count, 1)),
            "mean_abs_residual_mm": float(np.average(overall["mean_abs_residual_mm"], weights=overall["sample_count"])),
            "p90_abs_residual_mm": float(np.average(overall["p90_abs_residual_mm"], weights=overall["sample_count"])),
        }
    )
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    _, y, stage, baseline, residual, features = _training_state()

    direct_cv = q4_prediction._direct_cv(features, y, stage)
    rows = [
        {
            "model": "global_disturbance_direct_CV",
            "validation_scope": "time_block_cv",
            "RMSE_mm": direct_cv["global_direct_RMSE_mm"],
            "MAE_mm": direct_cv["global_direct_MAE_mm"],
        },
        {
            "model": "staged_disturbance_direct_CV",
            "validation_scope": "stage_time_block_cv",
            "RMSE_mm": direct_cv["staged_direct_RMSE_mm"],
            "MAE_mm": direct_cv["staged_direct_MAE_mm"],
        },
        {
            "model": "stage_normalized_baseline",
            "validation_scope": "training_reconstruction",
            "RMSE_mm": mean_squared_error(y, baseline) ** 0.5,
            "MAE_mm": mean_absolute_error(y, baseline),
            "monotonic_violations": monotonic_violations(baseline),
        },
    ]
    for name, factory in [
        ("baseline_plus_linear_residual", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("baseline_plus_gbdt_residual", q4_prediction._gbdt),
    ]:
        pred = np.zeros_like(y)
        for label in sorted(np.unique(stage)):
            idx = np.where(stage == label)[0]
            model = factory()
            model.fit(features.iloc[idx], residual[idx])
            pred[idx] = baseline[idx] + model.predict(features.iloc[idx])
        rows.append(
            {
                "model": name,
                "validation_scope": "stagewise_training_reconstruction",
                "RMSE_mm": mean_squared_error(y, pred) ** 0.5,
                "MAE_mm": mean_absolute_error(y, pred),
                "monotonic_violations": monotonic_violations(pred),
            }
        )
        cv_pred = np.full_like(y, np.nan, dtype=float)
        for label in sorted(np.unique(stage)):
            idx = np.where(stage == label)[0]
            for local_test in time_block_folds(len(idx), k=3):
                test_idx = idx[local_test]
                train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
                model = factory()
                model.fit(features.iloc[train_idx], residual[train_idx])
                cv_pred[test_idx] = baseline[test_idx] + model.predict(features.iloc[test_idx])
        mask = np.isfinite(cv_pred)
        projected = q4_prediction._project_stagewise_monotone(cv_pred, lower_bound=baseline, stage=stage)
        projected_mask = np.isfinite(projected)
        rows.append(
            {
                "model": f"{name}_CV",
                "validation_scope": "stage_residual_time_block_cv",
                "raw_RMSE_mm": mean_squared_error(y[mask], cv_pred[mask]) ** 0.5,
                "raw_MAE_mm": mean_absolute_error(y[mask], cv_pred[mask]),
                "raw_monotonic_violations": monotonic_violations(cv_pred[mask]),
                "projected_RMSE_mm": mean_squared_error(y[projected_mask], projected[projected_mask]) ** 0.5,
                "projected_MAE_mm": mean_absolute_error(y[projected_mask], projected[projected_mask]),
                "projected_monotonic_violations": monotonic_violations(projected[projected_mask]),
                "RMSE_mm": mean_squared_error(y[projected_mask], projected[projected_mask]) ** 0.5,
                "MAE_mm": mean_absolute_error(y[projected_mask], projected[projected_mask]),
                "monotonic_violations": monotonic_violations(projected[projected_mask]),
            }
        )

    ablation = pd.DataFrame(rows).sort_values("RMSE_mm")
    write_csv(ablation, AWARD_DIR / "q4_ablation_comparison.csv")

    shrinkage = pd.DataFrame(
        [_stage_residual_cv(features, y, stage, baseline, residual, value) for value in AUDIT_CONFIG.q4_residual_shrinkage_grid]
    ).sort_values("RMSE_mm")
    best_rmse = float(shrinkage.iloc[0]["RMSE_mm"])
    shrinkage["within_5pct_best"] = shrinkage["RMSE_mm"] <= 1.05 * best_rmse
    write_csv(shrinkage, AWARD_DIR / "q4_residual_shrinkage_sensitivity.csv")

    target_table = pd.read_csv(TABLE_DIR / "q4_table_4_1_predictions.csv")
    interval_rows = []
    for _, row in target_table.iterrows():
        stage_label = int(row["阶段标签"])
        idx = np.where(stage == stage_label)[0]
        stage_resid = residual[idx]
        width = max(1.96 * np.std(stage_resid), np.percentile(np.abs(stage_resid), 90))
        pred = float(row["表面位移预测值_mm"])
        interval_rows.append(
            {
                "时间点": row["时间点"],
                "实验集行号": int(row["实验集行号"]),
                "阶段标签": stage_label,
                "point_prediction_mm": pred,
                "lower_95_mm": max(0.0, pred - width),
                "upper_95_mm": pred + width,
                "interval_half_width_mm": width,
            }
        )
    intervals = pd.DataFrame(interval_rows)
    write_csv(intervals, AWARD_DIR / "q4_table_4_1_prediction_intervals.csv")
    coverage = _prediction_interval_coverage(stage, residual)
    write_csv(coverage, AWARD_DIR / "q4_prediction_interval_coverage.csv")
    write_csv(coverage, TABLE_DIR / "q4_prediction_interval_coverage.csv")

    generalized = ablation[ablation["validation_scope"].str.contains("cv", case=False, na=False)]
    best_generalized = generalized.iloc[0] if not generalized.empty else ablation.iloc[0]
    main_shrinkage_path = TABLE_DIR / "q4_residual_shrinkage_cv.csv"
    main_shrinkage = 0.35
    if main_shrinkage_path.exists():
        main_shrinkage = float(pd.read_csv(main_shrinkage_path).iloc[0]["shrinkage"])
    current = shrinkage.loc[np.isclose(shrinkage["shrinkage"], main_shrinkage)].iloc[0]
    return {
        "best_ablation": str(best_generalized["model"]),
        "best_training_reconstruction": str(ablation.iloc[0]["model"]),
        "recommended_shrinkage": float(shrinkage.iloc[0]["shrinkage"]),
        "main_shrinkage": float(main_shrinkage),
        "main_shrinkage_rmse": float(current["RMSE_mm"]),
        "best_shrinkage_rmse": best_rmse,
        "main_shrinkage_within_5pct": bool(current["within_5pct_best"]),
        "target_interval_max_width": float(intervals["interval_half_width_mm"].max()),
        "interval_empirical_coverage": float(coverage.loc[coverage["stage"].eq("overall"), "empirical_coverage"].iloc[0]),
        "update_main_table": bool((current["RMSE_mm"] - best_rmse) / max(current["RMSE_mm"], 1e-12) > 0.08),
    }
