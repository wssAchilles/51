from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = CODE_DIR.parent
PROBLEM_DIR = CODE_DIR / "2026-51MCM-Problem C"
ATTACHMENT_DIR = PROBLEM_DIR / "C 附件(Attachment)"
OUTPUT_DIR = CODE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
AWARD_DIR = OUTPUT_DIR / "award"


RANDOM_SEED = 20260501


@dataclass(frozen=True)
class AuditConfig:
    """Centralized parameters used by reproducibility and award-level audits."""

    rng_seed: int = RANDOM_SEED
    q1_huber_delta_grid: tuple[float, ...] = (1.0, 1.2, 1.345, 1.6, 2.0)
    q1_bootstrap_repeats: int = 250
    q2_velocity_windows: tuple[int, ...] = (18, 36, 72)
    q2_weight_grid: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, 0.55)
    q3_continuous_threshold_grid: tuple[float, ...] = (3.5, 4.0, 4.5, 5.0, 5.5)
    q3_sparse_quantile_grid: tuple[float, ...] = (0.995, 0.997, 0.999)
    q4_residual_shrinkage_grid: tuple[float, ...] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    q5_warning_windows: tuple[int, ...] = (18, 36, 72)
    q5_inverse_velocity_window_steps: int = 144


AUDIT_CONFIG = AuditConfig()


ATTACHMENTS = {
    "q1": ATTACHMENT_DIR / "附件1：两组位移时序数据-问题1.xlsx",
    "q2": ATTACHMENT_DIR / "附件2：位移时序数据-问题2.xlsx",
    "q3": ATTACHMENT_DIR / "附件3：监测数据（训练集与实验集）-问题3.xlsx",
    "q4": ATTACHMENT_DIR / "附件4：监测数据（训练集与实验集）-问题4.xlsx",
    "q5": ATTACHMENT_DIR / "附件5：监测数据-问题5.xlsx",
}


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, AWARD_DIR):
        path.mkdir(parents=True, exist_ok=True)
