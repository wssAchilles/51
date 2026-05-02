from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from slope_warning.common.io import write_json
from slope_warning.common.metrics import mae, maxae, p95ae, rmse, time_block_folds
from slope_warning.config import AWARD_DIR, MODEL_DIR, PROJECT_DIR, TABLE_DIR


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def time_block_cv_affine(x: np.ndarray, y: np.ndarray, fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], k: int = 5) -> dict[str, float]:
    rows = []
    for test in time_block_folds(len(x), k):
        train = np.setdiff1d(np.arange(len(x)), test, assume_unique=True)
        beta = fit_fn(x[train], y[train])
        pred = beta[0] + beta[1] * x[test]
        rows.append(
            {
                "RMSE": rmse(y[test], pred),
                "MAE": mae(y[test], pred),
                "P95AE": p95ae(y[test], pred),
                "MaxAE": maxae(y[test], pred),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def weighted_median(values: list[int], weights: list[float]) -> int:
    order = np.argsort(values)
    vals = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cutoff = 0.5 * w.sum()
    return int(vals[np.searchsorted(np.cumsum(w), cutoff)])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_baseline() -> dict[str, object]:
    """Record a compact baseline manifest before robustness checks overwrite outputs."""
    lock_path = AWARD_DIR / "baseline_lock.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text(encoding="utf-8"))
    summary_path = MODEL_DIR / "all_model_summaries.json"
    baseline = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    table_files = sorted(path for path in TABLE_DIR.glob("q*_*.csv") if path.is_file())
    manifest = {
        "policy": "稳健优先：除非扩展检验明确更强，否则正文主结果维持不变。",
        "model_summary": baseline,
        "tables": {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size} for path in table_files},
        "main_pdf": None,
    }
    pdf_path = PROJECT_DIR / "main.pdf"
    if pdf_path.exists():
        locked_pdf = AWARD_DIR / "baseline_main.pdf"
        if not locked_pdf.exists():
            shutil.copy2(pdf_path, locked_pdf)
        manifest["main_pdf"] = {"sha256": file_sha256(pdf_path), "bytes": pdf_path.stat().st_size}
    write_json(manifest, lock_path)
    return manifest
