from __future__ import annotations

import numpy as np
import pandas as pd

from slope_warning.common.io import read_excel, write_csv, write_json, write_text
from slope_warning.common.plotting import save_segmentation_plot
from slope_warning.common.preprocessing import hampel_replace, rolling_median
from slope_warning.common.segmentation import fit_stage_polynomial, two_breaks_constant_mean, two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, FIGURE_DIR, MODEL_DIR, TABLE_DIR


START_TIME = pd.Timestamp("2024-05-04 00:00:00")


def _sse_constant(values: np.ndarray, bounds: list[int]) -> float:
    arr = np.asarray(values, dtype=float)
    total = 0.0
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = arr[start:end]
        total += float(np.sum((seg - np.mean(seg)) ** 2))
    return total


def run() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q2"])
    displacement = df["表面位移_mm"].to_numpy(float)
    raw_velocity = np.r_[np.nan, np.diff(displacement) * 6.0]
    raw_velocity[0] = np.nanmedian(raw_velocity[1:20])
    clean_velocity, jump_flags = hampel_replace(raw_velocity, window=37, threshold=5.0, center=True)
    velocity_6h = rolling_median(clean_velocity, window=36, center=False, min_periods=18)

    b1, b2, velocity_sse = two_breaks_constant_mean(velocity_6h, min_len=500, step=10, refine=120)
    t_hours = (df["编号"].to_numpy(float) - 1.0) / 6.0
    disp_b1, disp_b2, displacement_sse = two_breaks_piecewise_linear(t_hours, displacement, min_len=500, step=20, refine=120)

    bounds = [0, b1, b2, len(df)]
    stage_fits = fit_stage_polynomial(t_hours, displacement, bounds, max_degree=3)
    stage_rows = []
    for fit in stage_fits:
        row = fit.__dict__.copy()
        row["start_time"] = START_TIME + pd.to_timedelta((fit.start - 1) / 6.0, unit="h")
        row["end_time"] = START_TIME + pd.to_timedelta((fit.end - 1) / 6.0, unit="h")
        row["coefficients"] = ";".join(f"{v:.10g}" for v in fit.coefficients)
        stage_rows.append(row)

    break_df = pd.DataFrame(
        {
            "转换节点": ["缓慢匀速形变->加速形变", "加速形变->快速形变"],
            "编号": [b1 + 1, b2 + 1],
            "时间": [
                START_TIME + pd.to_timedelta(b1 / 6.0, unit="h"),
                START_TIME + pd.to_timedelta(b2 / 6.0, unit="h"),
            ],
            "节点处表面位移_mm": [displacement[b1], displacement[b2]],
            "节点处6h平滑速度_mm_h": [velocity_6h[b1], velocity_6h[b2]],
        }
    )
    stage_df = pd.DataFrame(stage_rows)
    jump_df = pd.DataFrame(
        {
            "编号": df["编号"],
            "表面位移_mm": displacement,
            "原始速度_mm_h": raw_velocity,
            "Hampel瞬时跳变标记": jump_flags,
            "清洗后速度_mm_h": clean_velocity,
            "6h平滑速度_mm_h": velocity_6h,
        }
    )

    summary = {
        "method": "Hampel velocity cleaning + 6h trailing median velocity + two-break constant-mean dynamic programming",
        "break_indices_0_based": [int(b1), int(b2)],
        "break_serial_numbers": [int(b1 + 1), int(b2 + 1)],
        "break_times": [str(v) for v in break_df["时间"]],
        "velocity_segmentation_sse": float(velocity_sse),
        "displacement_piecewise_linear_breaks_for_validation": [int(disp_b1 + 1), int(disp_b2 + 1)],
        "displacement_piecewise_linear_sse_for_validation": float(displacement_sse),
        "velocity_sse_all": _sse_constant(velocity_6h, [0, len(df)]),
        "velocity_sse_three_stage": _sse_constant(velocity_6h, bounds),
        "instant_jump_count": int(jump_flags.sum()),
        "stage_models": stage_rows,
        "core_criteria": [
            "A real transition must persist beyond one smoothing window and shift the velocity level.",
            "The post-node smoothed acceleration trend cannot immediately revert to the prior level.",
            "The three-stage segmented cost must drop sharply versus one-stage velocity cost.",
            "Single-point Hampel velocity spikes are treated as noise/engineering disturbance, not stage transition.",
        ],
    }

    time = pd.Series(START_TIME + pd.to_timedelta(t_hours, unit="h"))
    save_segmentation_plot(time, displacement, velocity_6h, [b1, b2], FIGURE_DIR / "q2_segmentation.png", "Q2 velocity-driven three-stage segmentation")
    write_csv(break_df, TABLE_DIR / "q2_transition_nodes.csv")
    write_csv(stage_df, TABLE_DIR / "q2_stage_models.csv")
    write_csv(jump_df, TABLE_DIR / "q2_velocity_hampel_diagnostics.csv")
    write_json(summary, MODEL_DIR / "q2_model_summary.json")
    write_text(
        "\n".join(f"{item}" for item in summary["core_criteria"]),
        MODEL_DIR / "q2_noise_vs_transition_criteria.txt",
    )
    return summary

