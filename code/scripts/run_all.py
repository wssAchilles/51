from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slope_warning.config import MODEL_DIR, OUTPUT_DIR, ensure_output_dirs
from slope_warning.common.io import write_json
from slope_warning.questions import q1_calibration, q2_segmentation, q3_fusion, q4_prediction, q5_warning


def main() -> None:
    ensure_output_dirs()
    results = {}
    for name, runner in [
        ("q1", q1_calibration.run),
        ("q2", q2_segmentation.run),
        ("q3", q3_fusion.run),
        ("q4", q4_prediction.run),
        ("q5", q5_warning.run),
    ]:
        print(f"Running {name}...", flush=True)
        results[name] = runner()
        print(f"Finished {name}.", flush=True)
    write_json(results, MODEL_DIR / "all_model_summaries.json")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "questions": list(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

