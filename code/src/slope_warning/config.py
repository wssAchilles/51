from __future__ import annotations

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
