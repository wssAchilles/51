from __future__ import annotations

import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slope_warning.award_audit import run
from slope_warning.config import AWARD_DIR


def main() -> None:
    results = run()
    print(json.dumps({"output_dir": str(AWARD_DIR), "sections": list(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
