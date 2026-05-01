from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentFit:
    stage: int
    start: int
    end: int
    duration_h: float
    displacement_start: float
    displacement_end: float
    avg_velocity: float
    model: str
    degree: int
    coefficients: list[float]
    rmse: float
    r2: float
    bic: float


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.asarray(values, dtype=float))]


def two_breaks_constant_mean(values: np.ndarray, min_len: int = 500, step: int = 10, refine: int = 120) -> tuple[int, int, float]:
    y = np.asarray(values, dtype=float)
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    n = len(y)
    sy = _prefix(y)
    syy = _prefix(y * y)

    def sse(i: int, j: int) -> float:
        m = j - i
        s1 = sy[j] - sy[i]
        s2 = syy[j] - syy[i]
        return float(s2 - s1 * s1 / m)

    best = (float("inf"), min_len, n - min_len)
    for b1 in range(min_len, n - 2 * min_len, step):
        for b2 in range(b1 + min_len, n - min_len, step):
            val = sse(0, b1) + sse(b1, b2) + sse(b2, n)
            if val < best[0]:
                best = (val, b1, b2)

    _, r1, r2 = best
    best = (float("inf"), r1, r2)
    lo1, hi1 = max(min_len, r1 - refine), min(n - 2 * min_len, r1 + refine)
    for b1 in range(lo1, hi1 + 1):
        lo2, hi2 = max(b1 + min_len, r2 - refine), min(n - min_len, r2 + refine)
        for b2 in range(lo2, hi2 + 1):
            val = sse(0, b1) + sse(b1, b2) + sse(b2, n)
            if val < best[0]:
                best = (val, b1, b2)
    return best[1], best[2], best[0]


def two_breaks_piecewise_linear(t: np.ndarray, y: np.ndarray, min_len: int = 500, step: int = 10, refine: int = 120) -> tuple[int, int, float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    sx, sy, sxx, sxy, syy = (_prefix(v) for v in (t, y, t * t, t * y, y * y))

    def sse(i: int, j: int) -> float:
        m = j - i
        x1 = sx[j] - sx[i]
        y1 = sy[j] - sy[i]
        xx = sxx[j] - sxx[i]
        xy = sxy[j] - sxy[i]
        yy = syy[j] - syy[i]
        den = m * xx - x1 * x1
        if m < 2 or abs(den) < 1e-12:
            return 0.0
        b = (m * xy - x1 * y1) / den
        a = (y1 - b * x1) / m
        return float(yy + m * a * a + b * b * xx + 2 * a * b * x1 - 2 * a * y1 - 2 * b * xy)

    best = (float("inf"), min_len, n - min_len)
    for b1 in range(min_len, n - 2 * min_len, step):
        for b2 in range(b1 + min_len, n - min_len, step):
            val = sse(0, b1) + sse(b1, b2) + sse(b2, n)
            if val < best[0]:
                best = (val, b1, b2)

    _, r1, r2 = best
    best = (float("inf"), r1, r2)
    for b1 in range(max(min_len, r1 - refine), min(n - 2 * min_len, r1 + refine) + 1):
        for b2 in range(max(b1 + min_len, r2 - refine), min(n - min_len, r2 + refine) + 1):
            val = sse(0, b1) + sse(b1, b2) + sse(b2, n)
            if val < best[0]:
                best = (val, b1, b2)
    return best[1], best[2], best[0]


def fit_stage_polynomial(t_hours: np.ndarray, y: np.ndarray, bounds: list[int], max_degree: int = 3) -> list[SegmentFit]:
    fits: list[SegmentFit] = []
    for stage, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        x = np.asarray(t_hours[start:end], dtype=float)
        yy = np.asarray(y[start:end], dtype=float)
        x0 = x - x[0]
        best: SegmentFit | None = None
        for degree in range(1, max_degree + 1):
            coef = np.polyfit(x0, yy, degree)
            pred = np.polyval(coef, x0)
            resid = yy - pred
            sse = float(np.sum(resid * resid))
            n = len(yy)
            k = degree + 1
            bic = n * np.log(max(sse / n, 1e-12)) + k * np.log(n)
            rmse = float(np.sqrt(np.mean(resid * resid)))
            ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
            duration = (end - start - 1) / 6.0
            avg_v = (yy[-1] - yy[0]) / duration if duration > 0 else float("nan")
            candidate = SegmentFit(
                stage=stage,
                start=start + 1,
                end=end,
                duration_h=float(duration),
                displacement_start=float(yy[0]),
                displacement_end=float(yy[-1]),
                avg_velocity=float(avg_v),
                model=f"degree_{degree}_polynomial",
                degree=degree,
                coefficients=[float(v) for v in coef],
                rmse=rmse,
                r2=float(r2),
                bic=float(bic),
            )
            if best is None or candidate.bic < best.bic:
                best = candidate
        assert best is not None
        fits.append(best)
    return fits

