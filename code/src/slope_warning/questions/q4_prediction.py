from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from slope_warning.common.features import make_disturbance_features
from slope_warning.common.io import read_excel, write_csv, write_json
from slope_warning.common.metrics import time_block_folds
from slope_warning.common.plotting import save_prediction_plot, save_segmentation_plot
from slope_warning.common.preprocessing import hampel_replace, rolling_median
from slope_warning.common.segmentation import fit_stage_polynomial, two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, FIGURE_DIR, MODEL_DIR, TABLE_DIR


TARGET_TIMES = pd.to_datetime(
    ["2025-05-09 12:00", "2025-05-27 08:00", "2025-06-01 12:00", "2025-06-03 22:00", "2025-06-04 01:40"]
)


def _stage_from_breaks(n: int, b1: int, b2: int) -> np.ndarray:
    stage = np.ones(n, dtype=int)
    stage[b1:b2] = 2
    stage[b2:] = 3
    return stage


def _stage_tau(stage: np.ndarray) -> np.ndarray:
    tau = np.zeros(len(stage), dtype=float)
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        if len(idx) == 1:
            tau[idx] = 0.0
        else:
            tau[idx] = np.linspace(0.0, 1.0, len(idx))
    return tau


def _fit_stage_baselines(y: np.ndarray, stage: np.ndarray) -> dict[int, dict[str, object]]:
    baselines: dict[int, dict[str, object]] = {}
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        rel = y[idx] - y[idx[0]]
        tau = np.linspace(0.0, 1.0, len(idx))
        knots = np.linspace(0.0, 1.0, 15)
        values = np.interp(knots, tau, rel)
        values[0] = 0.0
        values[-1] = rel[-1]
        values = np.maximum.accumulate(values)
        if rel[-1] > 0:
            # Avoid zero-slope plateaus in the rapid stage caused by local sensor dips.
            # The rescaling keeps endpoints exact while preserving monotone creep growth.
            eps_ramp = np.linspace(0.0, 0.002 * rel[-1] * (len(values) - 1), len(values))
            values = values + eps_ramp
            values = (values - values[0]) / (values[-1] - values[0]) * rel[-1]
        baselines[int(label)] = {
            "interpolator": PchipInterpolator(knots, values, extrapolate=True),
            "increment": float(rel[-1]),
            "knots": knots,
            "values": values,
        }
    return baselines


def _baseline_values(stage: np.ndarray, baselines: dict[int, dict[str, object]]) -> np.ndarray:
    out = np.zeros(len(stage), dtype=float)
    offset = 0.0
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        tau = np.linspace(0.0, 1.0, len(idx))
        interp = baselines[int(label)]["interpolator"]
        rel = np.asarray(interp(tau), dtype=float)
        rel = rel - rel[0]
        rel = rel + (float(baselines[int(label)]["increment"]) - rel[-1]) * tau
        rel = np.maximum.accumulate(rel)
        out[idx] = offset + rel
        offset = out[idx[-1]]
    return out


def _gbdt() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        random_state=20260501,
        n_estimators=350,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=12,
        subsample=0.85,
    )


def _direct_cv(features: pd.DataFrame, y: np.ndarray, stage: np.ndarray) -> dict[str, float]:
    global_err = []
    for test_idx in time_block_folds(len(y), 5):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        model = _gbdt()
        model.fit(features.iloc[train_idx], y[train_idx])
        pred = model.predict(features.iloc[test_idx])
        global_err.append((mean_squared_error(y[test_idx], pred) ** 0.5, mean_absolute_error(y[test_idx], pred)))

    staged_pred = np.full(len(y), np.nan)
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        for test_idx_local in time_block_folds(len(idx), 3):
            test_idx = idx[test_idx_local]
            train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
            model = _gbdt()
            model.fit(features.iloc[train_idx], y[train_idx])
            staged_pred[test_idx] = model.predict(features.iloc[test_idx])
    mask = np.isfinite(staged_pred)
    return {
        "global_direct_RMSE_mm": float(np.mean([v[0] for v in global_err])),
        "global_direct_MAE_mm": float(np.mean([v[1] for v in global_err])),
        "staged_direct_RMSE_mm": float(mean_squared_error(y[mask], staged_pred[mask]) ** 0.5),
        "staged_direct_MAE_mm": float(mean_absolute_error(y[mask], staged_pred[mask])),
    }


def run() -> dict[str, object]:
    train = read_excel(ATTACHMENTS["q4"], sheet_name="训练集")
    exp = read_excel(ATTACHMENTS["q4"], sheet_name="实验集")
    train["时间"] = pd.to_datetime(train["时间"])
    exp["时间"] = pd.to_datetime(exp["时间"])
    y = train["表面位移_mm"].to_numpy(float)
    t_hours = (train["时间"] - train["时间"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    b1, b2, _ = two_breaks_piecewise_linear(t_hours, y, min_len=500, step=20, refine=120)
    stage_train = _stage_from_breaks(len(train), b1, b2)
    tau_train = _stage_tau(stage_train)
    stage_exp = exp["阶段标签"].to_numpy(int)
    tau_exp = _stage_tau(stage_exp)

    baselines = _fit_stage_baselines(y, stage_train)
    train_baseline = _baseline_values(stage_train, baselines)
    exp_baseline = _baseline_values(stage_exp, baselines)
    train_residual = y - train_baseline

    train_features, groups = make_disturbance_features(train, stage=stage_train, tau=tau_train)
    exp_features, _ = make_disturbance_features(exp, stage=stage_exp, tau=tau_exp)

    direct_cv = _direct_cv(train_features, y, stage_train)

    exp_pred = exp_baseline.copy()
    residual_models = {}
    residual_metrics = []
    for label in sorted(np.unique(stage_train)):
        train_idx = np.where(stage_train == label)[0]
        exp_idx = np.where(stage_exp == label)[0]
        model = _gbdt()
        model.fit(train_features.iloc[train_idx], train_residual[train_idx])
        train_resid_pred = model.predict(train_features.iloc[train_idx])
        residual = model.predict(exp_features.iloc[exp_idx])
        lo, hi = np.percentile(train_residual[train_idx], [2, 98])
        residual = np.clip(residual, lo, hi)
        current_offset = exp_pred[exp_idx[0] - 1] if exp_idx[0] > 0 else 0.0
        baseline_stage = exp_baseline[exp_idx] - exp_baseline[exp_idx][0] + current_offset
        residual_stage = exp_baseline[exp_idx] + residual
        residual_stage = residual_stage - residual_stage[0] + current_offset
        # The phase-normalized deformation curve is the physical backbone; residual learning
        # corrects disturbance bias but must not erase the stage's monotone creep growth.
        stage_pred = baseline_stage + 0.35 * (residual_stage - baseline_stage)
        stage_pred = np.maximum.accumulate(np.maximum(stage_pred, baseline_stage))
        exp_pred[exp_idx] = stage_pred
        residual_models[int(label)] = {
            "residual_clip_low": float(lo),
            "residual_clip_high": float(hi),
            "feature_importance": dict(zip(train_features.columns, model.feature_importances_.astype(float))),
        }
        residual_metrics.append(
            {
                "stage": int(label),
                "residual_RMSE_mm": float(mean_squared_error(train_residual[train_idx], train_resid_pred) ** 0.5),
                "residual_MAE_mm": float(mean_absolute_error(train_residual[train_idx], train_resid_pred)),
            }
        )

    exp_pred = np.maximum.accumulate(exp_pred)
    target_rows = []
    for target_time in TARGET_TIMES:
        matches = exp.index[exp["时间"].eq(target_time)].to_list()
        if not matches:
            raise ValueError(f"Target time not found in Q4 experiment set: {target_time}")
        row = matches[0]
        target_rows.append(
            {
                "时间点": target_time,
                "实验集行号": int(row + 1),
                "阶段标签": int(stage_exp[row]),
                "表面位移预测值_mm": float(exp_pred[row]),
            }
        )

    raw_velocity = np.r_[np.nan, np.diff(y) * 6.0]
    raw_velocity[0] = np.nanmedian(raw_velocity[1:20])
    clean_velocity, _ = hampel_replace(raw_velocity, window=37, threshold=5.0, center=True)
    velocity_6h = rolling_median(clean_velocity, 36, center=False, min_periods=18)
    stage_models = fit_stage_polynomial(t_hours, y, [0, b1, b2, len(y)], max_degree=3)

    transition_df = pd.DataFrame(
        {
            "转换节点": ["缓慢匀速形变->加速形变", "加速形变->快速形变"],
            "训练集行号": [b1 + 1, b2 + 1],
            "时间": [train["时间"].iloc[b1], train["时间"].iloc[b2]],
            "表面位移_mm": [y[b1], y[b2]],
            "6h速度_mm_h": [velocity_6h[b1], velocity_6h[b2]],
        }
    )
    stage_model_df = pd.DataFrame([{**fit.__dict__, "coefficients": ";".join(f"{v:.10g}" for v in fit.coefficients)} for fit in stage_models])
    baseline_df = pd.DataFrame(
        {
            "stage": list(baselines.keys()),
            "increment_mm": [baselines[k]["increment"] for k in baselines],
            "knots_tau": [";".join(f"{v:.6f}" for v in baselines[k]["knots"]) for k in baselines],
            "knots_displacement_mm": [";".join(f"{v:.6f}" for v in baselines[k]["values"]) for k in baselines],
        }
    )
    exp_result = exp.copy()
    exp_result["表面位移预测值_mm"] = exp_pred
    exp_result["阶段内归一化时间tau"] = tau_exp
    target_df = pd.DataFrame(target_rows)
    residual_metric_df = pd.DataFrame(residual_metrics)

    summary = {
        "training_transition_rows": [int(b1 + 1), int(b2 + 1)],
        "training_transition_times": [str(train["时间"].iloc[b1]), str(train["时间"].iloc[b2])],
        "direct_cv": direct_cv,
        "monotonic_violations_experiment": int(np.sum(np.diff(exp_pred) < -1e-9)),
        "target_predictions": target_rows,
        "residual_models": residual_models,
        "groups": groups,
    }

    write_csv(transition_df, TABLE_DIR / "q4_training_transition_nodes.csv")
    write_csv(stage_model_df, TABLE_DIR / "q4_training_stage_models.csv")
    write_csv(baseline_df, TABLE_DIR / "q4_stage_baseline_knots.csv")
    write_csv(residual_metric_df, TABLE_DIR / "q4_residual_model_metrics.csv")
    write_csv(exp_result, TABLE_DIR / "q4_experiment_surface_predictions.csv")
    write_csv(target_df, TABLE_DIR / "q4_table_4_1_predictions.csv")
    write_json(summary, MODEL_DIR / "q4_model_summary.json")
    save_segmentation_plot(train["时间"], y, velocity_6h, [b1, b2], FIGURE_DIR / "q4_training_segmentation.png", "Q4 training stage segmentation")
    save_prediction_plot(exp["时间"], exp_pred, FIGURE_DIR / "q4_experiment_prediction_curve.png", "Q4 experiment surface displacement prediction")
    save_prediction_plot(train["时间"], train_baseline, FIGURE_DIR / "q4_training_stage_baseline.png", "Q4 training stage-normalized baseline", observed=y)
    return summary
