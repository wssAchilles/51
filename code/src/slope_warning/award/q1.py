from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.linear_model import LinearRegression, RANSACRegressor

from slope_warning.award.common import rng, time_block_cv_affine
from slope_warning.common.io import read_excel, write_csv
from slope_warning.common.metrics import mae, maxae, p95ae, rmse
from slope_warning.common.plotting import configure_chinese_fonts
from slope_warning.config import ATTACHMENTS, AUDIT_CONFIG, AWARD_DIR
from slope_warning.questions import q1_calibration


configure_chinese_fonts()


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]


def _theilsen_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope, intercept, _, _ = theilslopes(y, x)
    return np.array([intercept, slope], dtype=float)


def _ransac_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = RANSACRegressor(
        estimator=LinearRegression(),
        random_state=AUDIT_CONFIG.rng_seed,
        min_samples=0.5,
        residual_threshold=5.0,
    )
    model.fit(x.reshape(-1, 1), y)
    return np.array([model.estimator_.intercept_, model.estimator_.coef_[0]], dtype=float)


def _save_diagnostics(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> None:
    pred = beta[0] + beta[1] * x
    residual = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(x, y, s=8, alpha=0.35)
    xs = np.linspace(x.min(), x.max(), 240)
    axes[0].plot(xs, beta[0] + beta[1] * xs, color="#c03a3a", lw=1.5)
    axes[0].set_xlabel("校正前数据A/mm")
    axes[0].set_ylabel("基准数据B/mm")
    axes[0].set_title("Huber稳健仿射校准")
    axes[0].grid(alpha=0.25)
    axes[1].hist(residual, bins=60, color="#2b6da8", alpha=0.78)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("校正残差/mm")
    axes[1].set_ylabel("频数")
    axes[1].set_title("残差分布")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q1_calibration_diagnostics.png", dpi=240)
    plt.close(fig)


def _residual_diagnostics(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> pd.DataFrame:
    pred = beta[0] + beta[1] * x
    residual = pred - y
    bins = pd.qcut(pred, q=5, duplicates="drop")
    by_bin = pd.DataFrame({"bin": bins, "residual": residual}).groupby("bin", observed=True)["residual"].apply(
        lambda s: float(np.sqrt(np.mean(np.square(s))))
    )
    return pd.DataFrame(
        [
            {
                "residual_mean_mm": float(np.mean(residual)),
                "residual_std_mm": float(np.std(residual)),
                "abs_residual_fitted_corr": float(np.corrcoef(np.abs(residual), pred)[0, 1]),
                "bin_rmse_ratio_max_min": float(by_bin.max() / max(by_bin.min(), 1e-12)),
                "train_RMSE_mm": rmse(y, pred),
                "train_MAE_mm": mae(y, pred),
                "train_P95AE_mm": p95ae(y, pred),
                "train_MaxAE_mm": maxae(y, pred),
            }
        ]
    )


def run() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q1"])
    x = df["数据A_光纤位移计数据_mm"].to_numpy(float)
    y = df["数据B_振弦式位移计数据_mm"].to_numpy(float)

    fits = {
        "OLS": _ols_fit,
        "Huber": q1_calibration.huber_affine_fit,
        "Theil-Sen": _theilsen_fit,
        "RANSAC": _ransac_fit,
    }
    rows = []
    for name, fit_fn in fits.items():
        beta = fit_fn(x, y)
        pred = beta[0] + beta[1] * x
        cv = time_block_cv_affine(x, y, fit_fn)
        rows.append(
            {
                "model": name,
                "beta0": beta[0],
                "beta1": beta[1],
                "train_RMSE": rmse(y, pred),
                "train_MAE": mae(y, pred),
                "train_P95AE": p95ae(y, pred),
                **{f"cv_{key}": value for key, value in cv.items()},
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["cv_MAE", "cv_RMSE"])
    write_csv(comparison, AWARD_DIR / "q1_model_comparison.csv")

    sensitivity_rows = []
    for delta_scale in AUDIT_CONFIG.q1_huber_delta_grid:
        fit_fn = lambda a, b, ds=delta_scale: q1_calibration.huber_affine_fit(a, b, delta_scale=ds)
        beta = fit_fn(x, y)
        pred = beta[0] + beta[1] * x
        cv = time_block_cv_affine(x, y, fit_fn)
        sensitivity_rows.append(
            {
                "delta_scale": delta_scale,
                "beta0": beta[0],
                "beta1": beta[1],
                "train_MAE": mae(y, pred),
                **{f"cv_{key}": value for key, value in cv.items()},
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(["cv_MAE", "cv_RMSE"])
    write_csv(sensitivity, AWARD_DIR / "q1_huber_delta_sensitivity.csv")

    boot = []
    generator = rng(AUDIT_CONFIG.rng_seed)
    for _ in range(AUDIT_CONFIG.q1_bootstrap_repeats):
        idx = generator.integers(0, len(x), len(x))
        beta = q1_calibration.huber_affine_fit(x[idx], y[idx])
        boot.append(beta[0] + beta[1] * q1_calibration.TARGET_VALUES)
    boot_arr = np.asarray(boot)
    main_beta = q1_calibration.huber_affine_fit(x, y)
    main_pred = main_beta[0] + main_beta[1] * q1_calibration.TARGET_VALUES
    ci = pd.DataFrame(
        {
            "校正前数据x": q1_calibration.TARGET_VALUES,
            "Huber校正值": main_pred,
            "bootstrap_CI_lower_2.5%": np.percentile(boot_arr, 2.5, axis=0),
            "bootstrap_CI_upper_97.5%": np.percentile(boot_arr, 97.5, axis=0),
            "bootstrap_std": boot_arr.std(axis=0),
        }
    )
    write_csv(ci, AWARD_DIR / "q1_bootstrap_correction_ci.csv")
    residual_diag = _residual_diagnostics(x, y, main_beta)
    write_csv(residual_diag, AWARD_DIR / "q1_residual_diagnostics.csv")
    _save_diagnostics(x, y, main_beta)

    huber_row = comparison.loc[comparison["model"].eq("Huber")].iloc[0]
    winner = comparison.iloc[0]
    return {
        "best_model": str(winner["model"]),
        "huber_cv_mae": float(huber_row["cv_MAE"]),
        "best_cv_mae": float(winner["cv_MAE"]),
        "recommended_delta_scale": float(sensitivity.iloc[0]["delta_scale"]),
        "ci_nan_count": int(ci.isna().sum().sum()),
        "heteroscedasticity_proxy": float(residual_diag.iloc[0]["abs_residual_fitted_corr"]),
        "update_main_table": bool(str(winner["model"]) != "Huber" and winner["cv_MAE"] < 0.98 * huber_row["cv_MAE"]),
    }
