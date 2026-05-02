from __future__ import annotations

from slope_warning.award.common import lock_baseline
from slope_warning.award.q1 import run as run_q1
from slope_warning.award.q2 import run as run_q2
from slope_warning.award.q3 import run as run_q3
from slope_warning.award.q4 import run as run_q4
from slope_warning.award.q5 import run as run_q5
from slope_warning.award.report import build_report
from slope_warning.common.io import write_json
from slope_warning.config import AWARD_DIR, ensure_output_dirs


def run() -> dict[str, object]:
    ensure_output_dirs()
    AWARD_DIR.mkdir(parents=True, exist_ok=True)
    baseline = lock_baseline()
    results = {
        "q1": run_q1(),
        "q2": run_q2(),
        "q3": run_q3(),
        "q4": run_q4(),
        "q5": run_q5(),
    }
    write_json(results, AWARD_DIR / "award_audit_summary.json")
    build_report(results, baseline)
    return results
