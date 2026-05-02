from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slope_warning.common.diagnostics import inverse_velocity_forecast
from slope_warning.common.io import write_csv
from slope_warning.common.plotting import configure_chinese_fonts
from slope_warning.common.preprocessing import rolling_median
from slope_warning.config import AUDIT_CONFIG, AWARD_DIR, TABLE_DIR


configure_chinese_fonts()


def _save_warning_timeline(pred: pd.DataFrame, inv: pd.DataFrame) -> None:
    thresholds = pd.read_csv(TABLE_DIR / "q5_warning_thresholds.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(pred["时间"], pred["表面位移_mm"], color="#2b6da8", lw=1.0, label="观测位移")
    axes[0].plot(pred["时间"], pred["模型积分位移_mm"], color="#c03a3a", lw=1.0, label="模型位移")
    axes[0].set_ylabel("表面位移/mm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(pred["时间"], pred["模型预测速度_mm_h"], color="#2e8b57", lw=0.9, label="预测速度")
    for _, row in thresholds.iterrows():
        axes[1].axhline(
            row["velocity_threshold_mm_h"],
            ls="--",
            lw=1,
            label=f"{row['level']} {row['velocity_threshold_mm_h']:.2f}",
        )
    if not inv.empty:
        axes[1].scatter(pd.to_datetime(inv["time"]), inv["velocity_mm_h"], color="#c03a3a", s=16, label="逆速度预警点")
    axes[1].set_ylabel("速度/(mm/h)")
    axes[1].set_xlabel("时间")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q5_warning_timeline.png", dpi=240)
    plt.close(fig)


def _warning_event_summary() -> pd.DataFrame:
    events_path = TABLE_DIR / "q5_warning_events.csv"
    if not events_path.exists():
        return pd.DataFrame(columns=["level", "event_count", "total_duration_h", "max_velocity_mm_h", "first_start_time"])
    events = pd.read_csv(events_path)
    if events.empty:
        return pd.DataFrame(columns=["level", "event_count", "total_duration_h", "max_velocity_mm_h", "first_start_time"])
    return (
        events.groupby("level", as_index=False)
        .agg(
            event_count=("level", "size"),
            total_duration_h=("duration_h", "sum"),
            max_velocity_mm_h=("max_velocity_mm_h", "max"),
            first_start_time=("start_time", "min"),
        )
        .sort_values("level")
    )


def _threshold_stability() -> pd.DataFrame:
    sens = pd.read_csv(TABLE_DIR / "q5_warning_window_sensitivity.csv")
    rows = []
    for stage, group in sens.dropna(subset=["q75", "q90", "q95"]).groupby("stage"):
        if str(stage).isdigit() or isinstance(stage, (int, float)):
            rows.append(
                {
                    "stage": int(stage),
                    "q75_range": float(group["q75"].max() - group["q75"].min()),
                    "q90_range": float(group["q90"].max() - group["q90"].min()),
                    "q95_range": float(group["q95"].max() - group["q95"].min()),
                    "window_count": int(group["window"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    combo = pd.read_csv(TABLE_DIR / "q5_variable_combination_cv.csv")
    combo["gbdt_displacement_rank"] = combo["gbdt_displacement_RMSE"].rank(method="min")
    combo["gbdt_velocity_rank"] = combo["gbdt_velocity_RMSE"].rank(method="min")
    combo["elasticnet_displacement_rank"] = combo["elasticnet_displacement_RMSE"].rank(method="min")
    combo["elasticnet_velocity_rank"] = combo["elasticnet_velocity_RMSE"].rank(method="min")
    combo["gbdt_stability_rank"] = combo["gbdt_displacement_rank"] + combo["gbdt_velocity_rank"]
    combo["elasticnet_stability_rank"] = combo["elasticnet_displacement_rank"] + combo["elasticnet_velocity_rank"]
    combo["consensus_rank_score"] = 0.65 * combo["gbdt_stability_rank"] + 0.35 * combo["elasticnet_stability_rank"]
    combo["model_rank_gap"] = (combo["gbdt_stability_rank"] - combo["elasticnet_stability_rank"]).abs()
    combo = combo.sort_values("consensus_rank_score")
    write_csv(combo, AWARD_DIR / "q5_variable_selection_stability.csv")
    conflict = combo[
        [
            "dropped_variable",
            "gbdt_stability_rank",
            "elasticnet_stability_rank",
            "model_rank_gap",
            "consensus_rank_score",
            "gbdt_displacement_RMSE",
            "elasticnet_displacement_RMSE",
        ]
    ].copy()
    conflict["interpretation"] = np.where(
        conflict["model_rank_gap"] <= 2,
        "两类模型排序接近，变量组合结论稳定。",
        "两类模型存在排序差异，需结合阶段非线性拟合和工程机理解释。",
    )
    write_csv(conflict, AWARD_DIR / "q5_consensus_conflict_analysis.csv")

    pred = pd.read_csv(TABLE_DIR / "q5_best_model_predictions.csv")
    pred["时间"] = pd.to_datetime(pred["时间"])
    inv = inverse_velocity_forecast(
        pred["时间"],
        pred["模型预测速度_mm_h"].to_numpy(float),
        stage=pred["阶段"].to_numpy(int),
        window_steps=AUDIT_CONFIG.q5_inverse_velocity_window_steps,
        min_velocity=0.3,
        min_r2=0.55,
    )
    write_csv(inv, AWARD_DIR / "q5_inverse_velocity_warning.csv")

    if inv.empty:
        lead_summary = pd.DataFrame(columns=["stage", "count", "lead_time_median_h", "lead_time_min_h", "lead_time_max_h"])
    else:
        lead_summary = (
            inv.groupby("stage", as_index=False)
            .agg(
                count=("lead_time_h", "size"),
                lead_time_median_h=("lead_time_h", "median"),
                lead_time_min_h=("lead_time_h", "min"),
                lead_time_max_h=("lead_time_h", "max"),
            )
            .sort_values("stage")
        )
    write_csv(lead_summary, AWARD_DIR / "q5_inverse_velocity_leadtime_summary.csv")

    event_summary = _warning_event_summary()
    write_csv(event_summary, AWARD_DIR / "q5_warning_event_summary.csv")
    threshold_stability = _threshold_stability()
    write_csv(threshold_stability, AWARD_DIR / "q5_threshold_stability.csv")

    pred["velocity_3h"] = rolling_median(pred["模型预测速度_mm_h"].to_numpy(float), 18, center=False, min_periods=9)
    pred["velocity_6h"] = rolling_median(pred["模型预测速度_mm_h"].to_numpy(float), 36, center=False, min_periods=18)
    pred["velocity_12h"] = rolling_median(pred["模型预测速度_mm_h"].to_numpy(float), 72, center=False, min_periods=36)
    write_csv(pred[["时间", "阶段", "velocity_3h", "velocity_6h", "velocity_12h"]], AWARD_DIR / "q5_warning_window_traces.csv")
    _save_warning_timeline(pred, inv)

    thresholds = pd.read_csv(TABLE_DIR / "q5_warning_thresholds.csv")
    return {
        "best_combo": str(combo.iloc[0]["variables"]),
        "inverse_warning_count": int(len(inv)),
        "event_levels": int(event_summary["level"].nunique()) if not event_summary.empty else 0,
        "thresholds_strictly_increasing": bool(np.all(np.diff(thresholds["velocity_threshold_mm_h"].to_numpy(float)) > 0)),
        "top_three_dropped_variables": combo.head(3)["dropped_variable"].astype(str).tolist(),
    }
