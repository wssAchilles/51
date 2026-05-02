from __future__ import annotations

import numpy as np
import pandas as pd

from slope_warning.award.common import weighted_median
from slope_warning.common.io import read_excel, write_csv
from slope_warning.common.preprocessing import hampel_replace, rolling_median
from slope_warning.common.segmentation import fit_stage_polynomial, two_breaks_constant_mean, two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, AUDIT_CONFIG, AWARD_DIR


def _first_sustained(values: np.ndarray, threshold: float, start: int = 0, span: int = 72, ratio: float = 0.8) -> int | None:
    above = np.asarray(values) > threshold
    for idx in range(start, len(above) - span):
        if above[idx : idx + span].mean() >= ratio:
            return idx
    return None


def _candidate_evidence() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    df = read_excel(ATTACHMENTS["q2"])
    y = df["表面位移_mm"].to_numpy(float)
    t_hours = (df["编号"].to_numpy(float) - 1.0) / 6.0
    start_time = pd.Timestamp("2024-05-04 00:00:00")
    raw_v = np.r_[np.nan, np.diff(y) * 6.0]
    raw_v[0] = np.nanmedian(raw_v[1:20])
    clean_v, _ = hampel_replace(raw_v, window=37, threshold=5.0, center=True)

    sensitivity_rows = []
    velocity_breaks = []
    for window in AUDIT_CONFIG.q2_velocity_windows:
        v_smooth = rolling_median(clean_v, window, center=False, min_periods=max(6, window // 2))
        b1, b2, sse = two_breaks_constant_mean(v_smooth, min_len=500, step=20, refine=120)
        velocity_breaks.append((b1, b2))
        sensitivity_rows.append(
            {
                "method": "velocity_level",
                "window_h": window / 6,
                "break1_serial": b1 + 1,
                "break2_serial": b2 + 1,
                "break1_time": start_time + pd.to_timedelta(b1 / 6, unit="h"),
                "break2_time": start_time + pd.to_timedelta(b2 / 6, unit="h"),
                "objective_sse": sse,
            }
        )
    disp_b1, disp_b2, disp_sse = two_breaks_piecewise_linear(t_hours, y, min_len=500, step=20, refine=120)
    sensitivity_rows.append(
        {
            "method": "displacement_piecewise_linear",
            "window_h": 0,
            "break1_serial": disp_b1 + 1,
            "break2_serial": disp_b2 + 1,
            "break1_time": start_time + pd.to_timedelta(disp_b1 / 6, unit="h"),
            "break2_time": start_time + pd.to_timedelta(disp_b2 / 6, unit="h"),
            "objective_sse": disp_sse,
        }
    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity, AWARD_DIR / "q2_transition_sensitivity.csv")

    v6 = rolling_median(clean_v, 36, center=False, min_periods=18)
    baseline = v6[:3000]
    med = np.median(baseline)
    mad = 1.4826 * np.median(np.abs(baseline - med))
    accel_onsets = []
    for mult in [3, 5, 8, 10]:
        threshold = med + mult * mad
        onset = _first_sustained(v6, threshold, span=72, ratio=0.8)
        accel_onsets.append((mult, threshold, onset))
    b1v, b2v = velocity_breaks[1]
    stage2_mean = v6[b1v:b2v].mean()
    stage3_mean = v6[b2v:].mean()
    rapid_threshold = (stage2_mean + stage3_mean) / 2.0
    rapid_onset = _first_sustained(v6, rapid_threshold, start=max(0, b1v - 72), span=36, ratio=0.8) or b2v

    candidate_rows = [
        {
            "transition": "slow_to_accelerated",
            "evidence": "displacement_trend_change",
            "serial": disp_b1 + 1,
            "weight": 0.20,
            "description": "分段位移拟合误差最优，反映累计趋势开始偏离。",
        },
        {
            "transition": "slow_to_accelerated",
            "evidence": "sustained_acceleration_onset",
            "serial": (accel_onsets[1][2] or b1v) + 1,
            "weight": 0.35,
            "description": "6h速度连续12h高于基线+5MAD，反映提前预警意义。",
        },
        {
            "transition": "slow_to_accelerated",
            "evidence": "velocity_level_jump",
            "serial": b1v + 1,
            "weight": 0.45,
            "description": "速度水平分段最优，反映阶段状态显著跃迁。",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "displacement_trend_change",
            "serial": disp_b2 + 1,
            "weight": 0.25,
            "description": "分段位移拟合误差最优。",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "rapid_velocity_persistence",
            "serial": rapid_onset + 1,
            "weight": 0.25,
            "description": "速度越过加速段与快速段中位水平并持续。",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "velocity_level_jump",
            "serial": b2v + 1,
            "weight": 0.50,
            "description": "速度水平分段最优。",
        },
    ]
    candidates = pd.DataFrame(candidate_rows)
    candidates["time"] = candidates["serial"].map(lambda s: start_time + pd.to_timedelta((s - 1) / 6, unit="h"))
    candidates["displacement_mm"] = candidates["serial"].map(lambda s: y[int(s) - 1])
    candidates["velocity_6h_mm_h"] = candidates["serial"].map(lambda s: v6[int(s) - 1])
    write_csv(candidates, AWARD_DIR / "q2_transition_candidate_comparison.csv")
    return candidates, sensitivity, y, t_hours, v6, start_time


def _weight_sensitivity(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for transition, group in candidates.groupby("transition"):
        values = group["serial"].astype(int).tolist()
        evidences = group["evidence"].tolist()
        for w1 in AUDIT_CONFIG.q2_weight_grid:
            for w2 in AUDIT_CONFIG.q2_weight_grid:
                for w3 in AUDIT_CONFIG.q2_weight_grid:
                    weights = np.asarray([w1, w2, w3], dtype=float)
                    weights = weights / weights.sum()
                    rows.append(
                        {
                            "transition": transition,
                            f"w_{evidences[0]}": weights[0],
                            f"w_{evidences[1]}": weights[1],
                            f"w_{evidences[2]}": weights[2],
                            "recommended_serial": weighted_median(values, weights.tolist()),
                        }
                    )
    out = pd.DataFrame(rows)
    write_csv(out, AWARD_DIR / "q2_weight_sensitivity.csv")
    return out


def run() -> dict[str, object]:
    candidates, _, y, t_hours, v6, start_time = _candidate_evidence()
    decision_rows = []
    for transition, group in candidates.groupby("transition"):
        recommended = weighted_median(group["serial"].astype(int).tolist(), group["weight"].astype(float).tolist())
        decision_rows.append(
            {
                "transition": transition,
                "recommended_serial": recommended,
                "recommended_time": start_time + pd.to_timedelta((recommended - 1) / 6, unit="h"),
                "recommended_displacement_mm": y[recommended - 1],
                "recommended_velocity_6h_mm_h": v6[recommended - 1],
                "supporting_evidence_count": int(len(group)),
                "note": "复合证据区分阶段起点、持续加速起点和速度显著跃迁点。",
            }
        )
    order = {"slow_to_accelerated": 1, "accelerated_to_rapid": 2}
    decision = pd.DataFrame(decision_rows).sort_values("transition", key=lambda s: s.map(order)).reset_index(drop=True)
    write_csv(decision, AWARD_DIR / "q2_final_transition_decision.csv")

    weight_sens = _weight_sensitivity(candidates)
    stability_rows = []
    for _, row in decision.iterrows():
        sub = weight_sens.loc[weight_sens["transition"].eq(row["transition"])]
        stability_rows.append(
            {
                "transition": row["transition"],
                "main_serial": int(row["recommended_serial"]),
                "same_serial_share": float(sub["recommended_serial"].eq(int(row["recommended_serial"])).mean()),
                "median_serial": float(sub["recommended_serial"].median()),
                "min_serial": int(sub["recommended_serial"].min()),
                "max_serial": int(sub["recommended_serial"].max()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    write_csv(stability, AWARD_DIR / "q2_transition_weight_stability_summary.csv")

    final_bounds = [0, int(decision.loc[0, "recommended_serial"]) - 1, int(decision.loc[1, "recommended_serial"]) - 1, len(y)]
    stage_rows = []
    for fit in fit_stage_polynomial(t_hours, y, final_bounds, max_degree=3):
        row = fit.__dict__.copy()
        row["start_time"] = start_time + pd.to_timedelta((fit.start - 1) / 6, unit="h")
        row["end_time"] = start_time + pd.to_timedelta((fit.end - 1) / 6, unit="h")
        row["coefficients"] = ";".join(f"{v:.10g}" for v in fit.coefficients)
        stage_rows.append(row)
    write_csv(pd.DataFrame(stage_rows), AWARD_DIR / "q2_final_stage_models.csv")
    return {
        "recommended": decision.to_dict(orient="records"),
        "weight_stability_min_share": float(stability["same_serial_share"].min()),
        "supporting_evidence_min_count": int(decision["supporting_evidence_count"].min()),
    }
