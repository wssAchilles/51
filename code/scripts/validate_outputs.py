from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slope_warning.config import AWARD_DIR, FIGURE_DIR, MODEL_DIR, TABLE_DIR


REQUIRED_TABLES = [
    "q1_table_1_1.csv",
    "q2_transition_nodes.csv",
    "q3_preprocessed_training.csv",
    "q3_preprocessed_experiment_features.csv",
    "q4_table_4_1_predictions.csv",
    "q4_experiment_surface_predictions.csv",
    "q5_variable_combination_cv.csv",
    "q5_warning_thresholds.csv",
    "q5_warning_events.csv",
]

REQUIRED_FIGURES = [
    "paper_problem_relationship.png",
    "paper_overall_model_route.png",
    "paper_preprocess_framework.png",
    "q2_composite_evidence_fusion.png",
    "q4_stage_residual_model.png",
    "q5_warning_closed_loop.png",
]

REQUIRED_AWARD = [
    "model_quality_report.md",
    "q1_model_comparison.csv",
    "q1_bootstrap_correction_ci.csv",
    "q2_final_transition_decision.csv",
    "q3_anomaly_sensitivity.csv",
    "q4_ablation_comparison.csv",
    "q4_prediction_interval_coverage.csv",
    "q4_table_4_1_prediction_intervals.csv",
    "q5_consensus_conflict_analysis.csv",
    "q5_variable_selection_stability.csv",
    "q5_inverse_velocity_warning.csv",
]


def _require_file(path: Path, errors: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing_or_empty:{path}")


def _assert_no_missing(df: pd.DataFrame, name: str, errors: list[str]) -> None:
    missing = int(df.isna().sum().sum())
    if missing:
        errors.append(f"{name}:missing_values={missing}")


def _assert_nonnegative(df: pd.DataFrame, name: str, columns: list[str], errors: list[str]) -> None:
    for column in columns:
        if column in df.columns and (df[column] < -1e-9).any():
            errors.append(f"{name}:{column}:negative_values")


def validate() -> dict[str, object]:
    errors: list[str] = []
    for filename in REQUIRED_TABLES:
        _require_file(TABLE_DIR / filename, errors)
    for filename in REQUIRED_FIGURES:
        _require_file(FIGURE_DIR / filename, errors)
    for filename in REQUIRED_AWARD:
        _require_file(AWARD_DIR / filename, errors)
    _require_file(MODEL_DIR / "all_model_summaries.json", errors)

    if (TABLE_DIR / "q3_preprocessed_training.csv").exists():
        q3_train = pd.read_csv(TABLE_DIR / "q3_preprocessed_training.csv")
        _assert_no_missing(q3_train, "q3_preprocessed_training", errors)
        _assert_nonnegative(q3_train, "q3_preprocessed_training", ["rain", "micro", "deep", "surface"], errors)

    if (TABLE_DIR / "q3_preprocessed_experiment_features.csv").exists():
        q3_exp = pd.read_csv(TABLE_DIR / "q3_preprocessed_experiment_features.csv")
        _assert_no_missing(q3_exp, "q3_preprocessed_experiment_features", errors)

    if (TABLE_DIR / "q4_experiment_surface_predictions.csv").exists():
        q4 = pd.read_csv(TABLE_DIR / "q4_experiment_surface_predictions.csv")
        col = "表面位移预测值_mm"
        if col not in q4.columns:
            errors.append("q4_experiment_surface_predictions:missing_prediction_column")
        elif int(np.sum(np.diff(q4[col].to_numpy(float)) < -1e-9)):
            errors.append("q4_experiment_surface_predictions:monotonicity_violation")

    if (TABLE_DIR / "q4_table_4_1_predictions.csv").exists():
        q4_table = pd.read_csv(TABLE_DIR / "q4_table_4_1_predictions.csv")
        if len(q4_table) != 5:
            errors.append(f"q4_table_4_1_predictions:target_count={len(q4_table)}")

    if (TABLE_DIR / "q5_warning_thresholds.csv").exists():
        thresholds = pd.read_csv(TABLE_DIR / "q5_warning_thresholds.csv")
        values = thresholds["velocity_threshold_mm_h"].to_numpy(float)
        if len(values) != 3 or not np.all(np.diff(values) > 0):
            errors.append("q5_warning_thresholds:not_strictly_increasing")

    if (AWARD_DIR / "q1_bootstrap_correction_ci.csv").exists():
        q1_ci = pd.read_csv(AWARD_DIR / "q1_bootstrap_correction_ci.csv")
        _assert_no_missing(q1_ci, "q1_bootstrap_correction_ci", errors)

    summary = {"ok": not errors, "error_count": len(errors), "errors": errors}
    (AWARD_DIR / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = validate()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
