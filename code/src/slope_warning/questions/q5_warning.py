from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slope_warning.common.features import five_variable_combinations, make_disturbance_features
from slope_warning.common.io import read_excel, write_csv, write_json
from slope_warning.common.metrics import time_block_folds
from slope_warning.common.plotting import save_prediction_plot, save_segmentation_plot
from slope_warning.common.preprocessing import hampel_replace, rolling_median
from slope_warning.common.segmentation import two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, FIGURE_DIR, MODEL_DIR, TABLE_DIR


def _stage_from_breaks(n: int, b1: int, b2: int) -> np.ndarray:
    stage = np.ones(n, dtype=int)
    stage[b1:b2] = 2
    stage[b2:] = 3
    return stage


def _stage_tau(stage: np.ndarray) -> np.ndarray:
    out = np.zeros(len(stage), dtype=float)
    for label in sorted(np.unique(stage)):
        idx = np.where(stage == label)[0]
        out[idx] = np.linspace(0.0, 1.0, len(idx)) if len(idx) > 1 else 0.0
    return out


def _gbdt() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        random_state=20260501,
        n_estimators=140,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=15,
        subsample=0.85,
    )


def _elastic() -> object:
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=20260501),
    )


def _evaluate_combo(features: pd.DataFrame, velocity_target: np.ndarray, displacement: np.ndarray) -> dict[str, float]:
    rows = []
    for model_name, model_factory in [("elasticnet", _elastic), ("gbdt", _gbdt)]:
        v_rmse, v_mae, d_rmse, end_err = [], [], [], []
        for test_idx in time_block_folds(len(velocity_target), 5, start=1):
            train_idx = np.setdiff1d(np.arange(1, len(velocity_target)), test_idx, assume_unique=True)
            model = model_factory()
            model.fit(features.iloc[train_idx], velocity_target[train_idx])
            pred_v = model.predict(features.iloc[test_idx])
            lo, hi = np.percentile(velocity_target[train_idx], [1, 99])
            pred_v = np.clip(pred_v, lo, hi)
            v_rmse.append(mean_squared_error(velocity_target[test_idx], pred_v) ** 0.5)
            v_mae.append(mean_absolute_error(velocity_target[test_idx], pred_v))
            pred_disp = displacement[test_idx[0]] + np.cumsum(pred_v) / 6.0
            actual_disp = displacement[test_idx]
            d_rmse.append(mean_squared_error(actual_disp, pred_disp) ** 0.5)
            end_err.append(abs(float(pred_disp[-1] - actual_disp[-1])))
        rows.extend(
            [
                (f"{model_name}_velocity_RMSE", np.mean(v_rmse)),
                (f"{model_name}_velocity_MAE", np.mean(v_mae)),
                (f"{model_name}_displacement_RMSE", np.mean(d_rmse)),
                (f"{model_name}_block_end_abs_error", np.mean(end_err)),
            ]
        )
    return {key: float(value) for key, value in rows}


def _sustained_events(time: pd.Series, values: np.ndarray, threshold: float, min_len: int) -> list[dict[str, object]]:
    above = np.asarray(values) >= threshold
    events = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= min_len:
                events.append(
                    {
                        "start_row": int(i + 1),
                        "end_row": int(j),
                        "start_time": str(time.iloc[i]),
                        "end_time": str(time.iloc[j - 1]),
                        "duration_h": float((j - i) / 6.0),
                        "max_velocity_mm_h": float(np.nanmax(values[i:j])),
                    }
                )
            i = j
        else:
            i += 1
    return events


def _warning_thresholds(time: pd.Series, velocity: np.ndarray, stage: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v6 = rolling_median(velocity, 36, center=False, min_periods=18)
    stats = []
    for label in sorted(np.unique(stage)):
        arr = v6[(stage == label) & np.isfinite(v6)]
        stats.append(
            {
                "stage": int(label),
                "q50": float(np.percentile(arr, 50)),
                "q75": float(np.percentile(arr, 75)),
                "q90": float(np.percentile(arr, 90)),
                "q95": float(np.percentile(arr, 95)),
            }
        )
    stat_df = pd.DataFrame(stats)
    stage1 = v6[(stage == 1) & np.isfinite(v6)]
    stage2 = v6[(stage == 2) & np.isfinite(v6)]
    stage3 = v6[(stage == 3) & np.isfinite(v6)]
    attention = max(float(np.percentile(stage1, 90)), float(np.percentile(stage2, 50)))
    warning = max(float(np.percentile(stage2, 90)), float(np.percentile(stage3, 10)))
    severe = max(float(np.percentile(stage3, 25)), warning + 0.5)
    thresholds = pd.DataFrame(
        [
            {
                "level": "关注",
                "velocity_threshold_mm_h": attention,
                "persistence_requirement": "连续不少于2小时",
                "extra_condition": "进入或接近加速阶段，或多源扰动残差持续为正",
            },
            {
                "level": "预警",
                "velocity_threshold_mm_h": warning,
                "persistence_requirement": "连续不少于2小时",
                "extra_condition": "逆速度序列连续下降，且降雨/孔压/微震至少一项增强",
            },
            {
                "level": "严重预警",
                "velocity_threshold_mm_h": severe,
                "persistence_requirement": "连续不少于6小时",
                "extra_condition": "逆速度线性外推失稳时间进入24小时窗口，或已处快速形变阶段",
            },
        ]
    )
    event_rows = []
    for _, row in thresholds.iterrows():
        min_len = 36 if row["level"] == "严重预警" else 12
        for event in _sustained_events(time, v6, float(row["velocity_threshold_mm_h"]), min_len):
            event_rows.append({"level": row["level"], **event})
    events = pd.DataFrame(event_rows)

    sens_rows = []
    for window, label in [(18, "3h"), (36, "6h"), (72, "12h")]:
        smoothed = rolling_median(velocity, window, center=False, min_periods=max(6, window // 2))
        for stage_label in sorted(np.unique(stage)):
            arr = smoothed[(stage == stage_label) & np.isfinite(smoothed)]
            sens_rows.append(
                {
                    "window": label,
                    "stage": int(stage_label),
                    "q75": float(np.percentile(arr, 75)),
                    "q90": float(np.percentile(arr, 90)),
                    "q95": float(np.percentile(arr, 95)),
                }
            )
    sensitivity = pd.DataFrame(sens_rows)
    return thresholds, events, pd.concat([stat_df.assign(window="6h_stage_stats"), sensitivity], ignore_index=True, sort=False)


def run() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q5"])
    df["时间"] = pd.to_datetime(df["时间"])
    displacement = df["表面位移_mm"].to_numpy(float)
    t_hours = (df["时间"] - df["时间"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    b1, b2, _ = two_breaks_piecewise_linear(t_hours, displacement, min_len=500, step=20, refine=120)
    stage = _stage_from_breaks(len(df), b1, b2)
    tau = _stage_tau(stage)

    raw_velocity = np.r_[np.nan, np.diff(displacement) * 6.0]
    raw_velocity[0] = np.nanmedian(raw_velocity[1:20])
    clean_velocity, velocity_flags = hampel_replace(raw_velocity, window=37, threshold=5.0, center=True)

    combo_rows = []
    for combo in five_variable_combinations():
        features, _ = make_disturbance_features(df, include=list(combo), stage=stage, tau=tau)
        metrics = _evaluate_combo(features, clean_velocity, displacement)
        combo_rows.append(
            {
                "variables": "+".join(combo),
                "dropped_variable": next(v for v in ["降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数", "爆破点距离_m", "单段最大药量_kg"] if v not in combo),
                "feature_count": int(features.shape[1]),
                **metrics,
            }
        )
    combo_df = pd.DataFrame(combo_rows)
    combo_df["gbdt_rank_score"] = combo_df["gbdt_displacement_RMSE"].rank(method="min") + 0.3 * combo_df["gbdt_velocity_RMSE"].rank(method="min")
    combo_df["elasticnet_rank_score"] = combo_df["elasticnet_displacement_RMSE"].rank(method="min") + 0.3 * combo_df["elasticnet_velocity_RMSE"].rank(method="min")
    combo_df = combo_df.sort_values(["gbdt_rank_score", "elasticnet_rank_score"])
    best_combo = tuple(combo_df.iloc[0]["variables"].split("+"))

    best_features, groups = make_disturbance_features(df, include=list(best_combo), stage=stage, tau=tau)
    best_model = _gbdt()
    best_model.fit(best_features, clean_velocity)
    pred_velocity = best_model.predict(best_features)
    pred_velocity = np.clip(pred_velocity, np.percentile(clean_velocity[1:], 1), np.percentile(clean_velocity[1:], 99))
    pred_displacement = displacement[0] + np.cumsum(pred_velocity) / 6.0
    pred_displacement = pred_displacement - pred_displacement[0] + displacement[0]

    thresholds, events, sensitivity = _warning_thresholds(df["时间"], pred_velocity, stage)
    stage_df = pd.DataFrame(
        [
            {
                "stage": int(label),
                "start_row": int(np.where(stage == label)[0][0] + 1),
                "end_row": int(np.where(stage == label)[0][-1] + 1),
                "start_time": str(df["时间"].iloc[np.where(stage == label)[0][0]]),
                "end_time": str(df["时间"].iloc[np.where(stage == label)[0][-1]]),
                "displacement_start_mm": float(displacement[np.where(stage == label)[0][0]]),
                "displacement_end_mm": float(displacement[np.where(stage == label)[0][-1]]),
                "avg_velocity_mm_h": float(
                    (displacement[np.where(stage == label)[0][-1]] - displacement[np.where(stage == label)[0][0]])
                    / ((len(np.where(stage == label)[0]) - 1) / 6.0)
                ),
            }
            for label in sorted(np.unique(stage))
        ]
    )
    prediction_df = df.copy()
    prediction_df["阶段"] = stage
    prediction_df["阶段内归一化时间tau"] = tau
    prediction_df["原始速度_mm_h"] = raw_velocity
    prediction_df["清洗后速度_mm_h"] = clean_velocity
    prediction_df["速度异常标记"] = velocity_flags
    prediction_df["模型预测速度_mm_h"] = pred_velocity
    prediction_df["模型积分位移_mm"] = pred_displacement

    feature_importance = pd.DataFrame({"feature": best_features.columns, "gbdt_importance": best_model.feature_importances_}).sort_values(
        "gbdt_importance", ascending=False
    )
    summary = {
        "stage_transition_rows": [int(b1 + 1), int(b2 + 1)],
        "stage_transition_times": [str(df["时间"].iloc[b1]), str(df["时间"].iloc[b2])],
        "best_combo": list(best_combo),
        "dropped_variable": str(combo_df.iloc[0]["dropped_variable"]),
        "best_combo_metrics": combo_df.iloc[0].to_dict(),
        "warning_thresholds": thresholds.to_dict(orient="records"),
        "groups": groups,
        "velocity_jump_count": int(velocity_flags.sum()),
    }

    write_csv(stage_df, TABLE_DIR / "q5_stage_division.csv")
    write_csv(combo_df, TABLE_DIR / "q5_variable_combination_cv.csv")
    write_csv(prediction_df, TABLE_DIR / "q5_best_model_predictions.csv")
    write_csv(feature_importance, TABLE_DIR / "q5_best_model_feature_importance.csv")
    write_csv(thresholds, TABLE_DIR / "q5_warning_thresholds.csv")
    write_csv(events, TABLE_DIR / "q5_warning_events.csv")
    write_csv(sensitivity, TABLE_DIR / "q5_warning_window_sensitivity.csv")
    write_json(summary, MODEL_DIR / "q5_model_summary.json")
    save_segmentation_plot(df["时间"], displacement, rolling_median(clean_velocity, 36, center=False, min_periods=18), [b1, b2], FIGURE_DIR / "q5_stage_segmentation.png", "Q5 stage division")
    save_prediction_plot(df["时间"], pred_displacement, FIGURE_DIR / "q5_best_model_displacement_fit.png", "Q5 best variable-combination model", observed=displacement)
    return summary
